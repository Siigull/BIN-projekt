import re
import graphviz

def parse_and_draw_cgp(cgp_string, output_filename="parity_circuit"):
    # 1. Strip the leading brace and split into Header, Nodes, and Output
    try:
        # This prevents the ValueError by removing the { before parsing integers
        clean_string = cgp_string.strip().lstrip('{')
        
        header_str = clean_string.split('}')[0]
        outputs_str = clean_string.split('(')[-1].replace(')', '')
        nodes_str = clean_string[clean_string.find('}')+1 : clean_string.rfind('(')]
    except IndexError:
        print("Error: String does not match expected CGP format.")
        return

    # 2. Extract parameters
    try:
        params = [int(x.strip()) for x in header_str.split(',')]
        num_inputs = params[0]
        outputs = [int(x.strip()) for x in outputs_str.split(',')]
    except ValueError as e:
        print(f"Error parsing parameters: {e}")
        return

    # 3. Parse the nodes using regex
    node_matches = re.findall(r'\[(\d+)\](\d+),(\d+),(\d+)', nodes_str)
    
    nodes = {}
    for m in node_matches:
        node_id = int(m[0])
        nodes[node_id] = {
            'in1': int(m[1]), 
            'in2': int(m[2]), 
            'func': int(m[3])
        }

    # 4. Trace the active path (backward from the outputs)
    active_nodes = set(outputs)
    to_process = list(outputs)
    
    while to_process:
        curr = to_process.pop(0)
        if curr in nodes:
            in1 = nodes[curr]['in1']
            in2 = nodes[curr]['in2']
            if in1 not in active_nodes:
                active_nodes.add(in1)
                to_process.append(in1)
            if in2 not in active_nodes:
                active_nodes.add(in2)
                to_process.append(in2)

    # 5. Build the visual graph
    dot = graphviz.Digraph(comment='CGP Circuit')
    dot.attr(rankdir='LR') 
    
    # Draw Inputs
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(label='Inputs')
        for i in range(num_inputs):
            color = 'blue' if i in active_nodes else 'lightgrey'
            c.node(str(i), f'In {i}', shape='invhouse', color=color, fontcolor=color)

    # Draw Logic Gates
    for node_id, data in nodes.items():
        is_active = node_id in active_nodes
        color = 'black' if is_active else 'lightgrey'
        
        label = f"ID:{node_id}\nFunc:{data['func']}"
        dot.node(str(node_id), label, shape='box', color=color, fontcolor=color)
        
        edge_color = 'black' if is_active else 'lightgrey'
        dot.edge(str(data['in1']), str(node_id), color=edge_color)
        dot.edge(str(data['in2']), str(node_id), color=edge_color)

    # Draw Outputs
    for i, out_id in enumerate(outputs):
        out_node_name = f'out_{i}'
        dot.node(out_node_name, f'Out {i}', shape='house', color='red')
        dot.edge(str(out_id), out_node_name, color='red', penwidth='2.0')

    # 6. Render the output file
    dot.render(output_filename, format='png', cleanup=True)
    print(f"Success! Graph saved as {output_filename}.png in your current directory.")

# Your exact string with the { included
cgp_str = "{5,1, 3,3, 2,3,4}([5]1,3,1)([6]4,0,3)([7]2,3,3)([8]1,7,3)([9]5,2,3)([10]1,1,1)([11]1,3,1)([12]6,8,3)([13]7,2,0)(12)"

parse_and_draw_cgp(cgp_str)

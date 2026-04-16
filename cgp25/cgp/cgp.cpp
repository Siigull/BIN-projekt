#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#if defined(__AVX2__)
    #include <immintrin.h>
    #define SIMD_WIDTH 8
    #define SIMD_TYPE __m256i
#elif defined(__ARM_NEON) || defined(__aarch64__)
    #include <arm_neon.h>
    #define SIMD_WIDTH 4
    #define SIMD_TYPE uint32x4_t
#else
    #error "No SIMD support"
#endif
#include <vector>
#include <algorithm>
#include "cgp.h"

typedef int *chromozom;              //dynamicke pole int, velikost dana m*n*(vstupu bloku+vystupu bloku) + vystupu komb
chromozom *populace[POPULACE_MAX];   //pole ukazatelu na chromozomy jedincu populace
chromozom* populace_arena = nullptr; //populace definovana za sebou kvůli mezipameti
int fitt[POPULACE_MAX];              //fitness jedincu populace
int uzitobloku(chromozom p_chrom);

int bestfit, bestfit_idx;   //nejlepsi fittnes, index jedince v populaci
int bestblk;

int *vystupy;               //pole vystupnich hodnot pro vyhodnocovani fce
int *pouzite;               //pole, kde kazda polozka odpovida bloku a urcuje zda se jedna o pouzity blok

int param_m = PARAM_M;             //pocet sloupcu
int param_n = PARAM_N;             //pocet radku
int param_in = PARAM_IN;           //pocet vstupu komb. obvodu
int param_out = PARAM_OUT;         //pocet vystupu komb. obvodu
int param_populace = POPULACE_MAX; //pocet jedincu populace
int block_in = 2;             //pocet vstupu  jednoho bloku (neni impl pro zmenu)
int l_back = L_BACK;              // 1 (pouze predchozi sloupec)  .. param_m (maximalni mozny rozsah);
#ifdef CHROMOZOME_LOAD
char chrin[] = CHROMOZOME_LOAD;
#else
char chrin = null;
#endif


int param_fitev;  //pocet pruchodu pro ohodnoceni jednoho chromozomu, vznikne jako (pocet vstupnich dat/(pocet vstupu+pocet vystupu))

int param_generaci; //pocet kroku evoluce
int last_improvement = 0;
int tdata_int[DATASIZE];

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
SIMD_TYPE* tdata;
SIMD_TYPE* valid_masks;

int sizesloupec = param_n*(block_in+1); //pocet polozek ktery zabira sloupec v chromozomu
int outputidx   = param_m*sizesloupec; //index v poli chromozomu, kde zacinaji vystupy
int maxidx_out  = param_n*param_m + param_in; //max. index pouzitelny jako vstup  pro vystupy
int maxfitness  = 0; //max. hodnota fitness

int fitpop, maxfitpop; //fitness populace

//Rozšíření pro 
double fitepsilon = FIT_EPSILON; //Splňuje cíl operace v epsilon zlomku populace = uspech
int fitnessepsilon = 0; //Nastaven v main(). Epsilon přepočítán na počet správných výsledků.
int maxblkfitness = PARAM_M * PARAM_N; //max. hodnota fitness obsahu obvodu

typedef struct { //struktura obsahujici mozne hodnoty vstupnich poli chromozomu pro urcity sloupec
    int pocet;   //pouziva se pri generovani noveho cisla pri mutaci
    int *hodnoty;
} sl_rndval;

sl_rndval **sloupce_val;  //predpocitane mozne hodnoty vstupu pro jednotlive sloupce
#define ARRSIZE PARAM_M*PARAM_N    //velikost pole = pocet bloku

#define LOOKUPTABSIZE 256
unsigned char lookupbit_tab[LOOKUPTABSIZE]; //LookUp pro rychle zjisteni poctu nastavenych bitu na 0 v 8bit cisle

#define copy_chromozome(from,to) (chromozom *) memcpy(to, from, (outputidx + param_out)*sizeof(int));

#define FITNESS_CALLCNT (POPULACE_MAX + PARAM_GENERATIONS*POPULACE_MAX) //pocet volani funkce fitness

//-----------------------------------------------------------------------
//Vypis chromozomu
//=======================================================================
//p_chrom ukazatel na chromozom
//-----------------------------------------------------------------------
void print_chrom(FILE *fout, chromozom p_chrom) {
  fprintf(fout, "{%d,%d, %d,%d, %d,%d,%d}", param_in, param_out, param_m, param_n, block_in, l_back, uzitobloku(p_chrom));
  for(int i=0; i<outputidx; i++) {
     if (i % 3 == 0) fprintf(fout,"([%d]",(i/3)+param_in);
     fprintf(fout,"%d", *p_chrom++);
     ((i+1) % 3 == 0) ? fprintf(fout,")") : fprintf(fout,",");
   }
  fprintf(fout,"(");
  for(int i=outputidx; i<outputidx+param_out; i++) {
     if (i > outputidx) fprintf(fout,",");
     fprintf(fout,"%d", *p_chrom++);
  }
  fprintf(fout,")");
  fprintf(fout,"\n");
}

void print_xls(FILE *xlsfil) {
  fprintf(xlsfil, "%d\t%d\t%d\t%d\t\t",param_generaci,bestfit,fitpop,bestblk);
  for (int i=0; i < param_populace;  i++)
      fprintf(xlsfil, "%d\t",fitt[i]);
  fprintf(xlsfil, "\n");
}

//-----------------------------------------------------------------------
//Nacteni chromozomu
//=======================================================================
//str string ve kterem je chromozom
//nstr delka str
//p_chrom ukazatel na chromozom
//-----------------------------------------------------------------------
bool load_chrom(char* str, size_t nstr, chromozom p_chrom){
#define NEXT() do {\
    if (++stri >= nstr) { return false; }\
} while(false);
#define SKIP_WHITE() do {\
    if (stri >= nstr) return false;\
    while (isspace(str[stri])) {\
        stri++;\
        if (stri >= nstr) return false;\
    }\
} while(false);
#define EXPECT(Mchr) do {\
    SKIP_WHITE();\
    if (str[stri] != (Mchr)) return false;\
    NEXT();\
} while(false);
#define READ_INT(num) do {\
    SKIP_WHITE();\
    num = strtol(str + stri, &endptr, 10);\
    if (endptr == (str + stri)) return false;\
    stri = endptr - str;\
} while(false);

    char* endptr;
    int stri=0, buf;
    EXPECT('{');
    READ_INT(buf); if(param_in != buf)  return false; EXPECT(',');
    READ_INT(buf); if(param_out != buf) return false; EXPECT(',');
    READ_INT(buf); if(param_m != buf)   return false; EXPECT(',');
    READ_INT(buf); if(param_n != buf)   return false; EXPECT(',');
    READ_INT(buf); if(block_in != buf)  return false; EXPECT(',');
    READ_INT(buf); if(l_back != buf)    return false; EXPECT(',');
    READ_INT(buf); // uzitobloku, počítáme funkcí, není caching
    EXPECT('}');

    for (int i=0; i < param_m; i++) {
        for (int j=0; j < param_n; j++) {
            EXPECT('(');
            EXPECT('[');
            READ_INT(buf); int node_start = (buf-param_in)*3;
            EXPECT(']');
            READ_INT(buf); p_chrom[node_start]     = buf; // in1
            EXPECT(',');
            READ_INT(buf); p_chrom[node_start + 1] = buf; // in2
            EXPECT(',');
            READ_INT(buf); p_chrom[node_start + 2] = buf; // func
            if (buf >= FUNCTIONS) return false;
            EXPECT(')');
        }
    }

    EXPECT('(');
    for (int i=0; i < param_out-1; i++) {
        READ_INT(buf); *(p_chrom + outputidx + i) = buf;
        EXPECT(',');
    }

    // Just because last doesn't have comma
    READ_INT(buf); *(p_chrom + outputidx + (param_out-1)) = buf;
    EXPECT(')');

    return true;

#undef NEXT
#undef SKIP_WHITE
#undef EXPECT
#undef READ_INT
}

//-----------------------------------------------------------------------
//POCET POUZITYCH BLOKU
//=======================================================================
//p_chrom ukazatel na chromozom,jenz se ma ohodnotit
//-----------------------------------------------------------------------
int uzitobloku(chromozom p_chrom) {
    int i,j, in,idx, poc = 0;
    int *p_pom;
    memset(pouzite, 0, maxidx_out*sizeof(int));

    //oznacit jako pouzite bloky napojene na vystupy
    p_pom = p_chrom + outputidx;
    for (i=0; i < param_out; i++) {
        in = *p_pom++;
        pouzite[in] = 1;
    }

    //pruchod od vystupu ke vstupum
    p_pom = p_chrom + outputidx - 1;
    idx = maxidx_out-1;
    for (i=param_m; i > 0; i--) {
        for (j=param_n; j > 0; j--,idx--) {
            p_pom--; //fce
            if (pouzite[idx] == 1) { //pokud je blok pouzit, oznacit jako pouzite i bloky, na ktere je napojen
               in = *p_pom--; //in2
               pouzite[in] = 1;
               in = *p_pom--; //in1
               pouzite[in] = 1;
               poc++;
            } else {
               p_pom -= block_in; //posun na predchozi blok
            }
        }
    }

    return poc;
}

inline int popcount_simd(SIMD_TYPE v) {
#if defined(__AVX2__)
    const SIMD_TYPE lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const SIMD_TYPE low_mask = _mm256_set1_epi8(0x0F);
    SIMD_TYPE lo = _mm256_and_si256(v, low_mask);
    SIMD_TYPE hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
    SIMD_TYPE popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    SIMD_TYPE popcnt2 = _mm256_shuffle_epi8(lookup, hi);
    SIMD_TYPE total = _mm256_add_epi8(popcnt1, popcnt2);
    total = _mm256_sad_epu8(total, _mm256_setzero_si256());
    return _mm256_extract_epi64(total, 0) + 
           _mm256_extract_epi64(total, 1) + 
           _mm256_extract_epi64(total, 2) + 
           _mm256_extract_epi64(total, 3);
#else  // NEON
    uint8x16_t counts = vcntq_u8(vreinterpretq_u8_u32(v));
    return vaddvq_u8(counts);
#endif
}

struct ActiveChrom {
    struct Node { int in1, in2, fce, out_idx; };
    std::vector<Node> nodes;  // Only active nodes, topologically sorted
    int out_indices[PARAM_OUT];
    int active_count;
};

ActiveChrom active_popul[POPULACE_MAX];

void log_active_nodes_by_column(FILE* fout, int pop_idx) {
    const ActiveChrom& ac = active_popul[pop_idx];
    std::vector<int> col_counts(param_m, 0);

    for (const auto& node : ac.nodes) {
        int flat_idx = node.out_idx - param_in;
        if (flat_idx < 0) continue;
        int col = flat_idx / param_n;
        if (col >= 0 && col < param_m) {
            col_counts[col]++;
        }
    }

    fprintf(fout, "%d: Active nodes by column:", param_generaci);
    for (int c = 0; c < param_m; c++) {
        fprintf(fout, "%d, ", col_counts[c]);
    }
    fprintf(fout, "\n");
}

void precompute_active(int popul_index) {
    ActiveChrom& ac = active_popul[popul_index];
    chromozom p_chrom = (chromozom)populace[popul_index];
    ac.nodes.clear();

    memset(pouzite, 0, maxidx_out*sizeof(int));

    int* p_pom = p_chrom + outputidx;
    for (int i = 0; i < param_out; i++) {
        int in = *p_pom++;
        ac.out_indices[i] = in;
        pouzite[in] = 1;
    }

    std::vector<ActiveChrom::Node> rev;
    p_pom = p_chrom + outputidx - 1;
    int idx_loop = maxidx_out-1;
    for (int i = param_m - 1; i >= 0; i--) {
        for (int j = param_n - 1; j >= 0; j--, idx_loop--) {
            if (pouzite[idx_loop]) {
                int fce = *p_pom--;
                int in2 = *p_pom--;
                pouzite[in2] = 1;
                int in1 = *p_pom--;
                pouzite[in1] = 1;

                int out_idx = (i * param_n + j) + param_in;
                rev.push_back({in1, in2, fce, out_idx});
            } else {
                p_pom -= (block_in + 1); // +1 for function
            }
        }
    }

    std::reverse(rev.begin(), rev.end());
    ac.nodes = std::move(rev);
    ac.active_count = (int)ac.nodes.size();
}

inline int fitness(int pop_idx, const SIMD_TYPE* __restrict p_svystup, SIMD_TYPE valid_mask) {
    const ActiveChrom& ac = active_popul[pop_idx];
    int i, j;
    SIMD_TYPE *v_vystupy = (SIMD_TYPE*)vystupy;
    SIMD_TYPE *p_v_vystup = v_vystupy + param_in;
    static const SIMD_TYPE ones = 
#if defined(__AVX2__)
        _mm256_set1_epi32(-1);
#else
        vdupq_n_u32((uint32_t)-1);
#endif
    
    for (const auto& node : ac.nodes) {
        SIMD_TYPE in1 = v_vystupy[node.in1];
        SIMD_TYPE in2 = v_vystupy[node.in2];
        SIMD_TYPE res;
        
#if defined(__AVX2__)
        if      (node.fce == 0) res = in1;
        else if (node.fce == 1) res = _mm256_and_si256(in1, in2);
#if FUNCTIONS >= 3
        else if (node.fce == 2) res = _mm256_or_si256(in1, in2);
#endif
#if FUNCTIONS >= 4
        else if (node.fce == 3) res = _mm256_xor_si256(in1, in2);
#endif
#if FUNCTIONS >= 5
        else if (node.fce == 4) res = _mm256_andnot_si256(in1, ones);
#endif
#if FUNCTIONS >= 6
        else if (node.fce == 5) res = _mm256_andnot_si256(in2, ones);
#endif
#if FUNCTIONS >= 7
        else if (node.fce == 6) res = _mm256_and_si256(in1, _mm256_andnot_si256(in2, ones));
#endif
#if FUNCTIONS >= 8
        else if (node.fce == 7) res = _mm256_xor_si256(_mm256_and_si256(in1, in2), ones);
#endif
#if FUNCTIONS >= 9
        else if (node.fce == 8) res = _mm256_xor_si256(_mm256_or_si256(in1, in2), ones);
#endif
#else  // NEON
        if      (node.fce == 0) res = in1;
        else if (node.fce == 1) res = vandq_u32(in1, in2);
#if FUNCTIONS >= 3
        else if (node.fce == 2) res = vorrq_u32(in1, in2);
#endif
#if FUNCTIONS >= 4
        else if (node.fce == 3) res = veorq_u32(in1, in2);
#endif
#if FUNCTIONS >= 5
        else if (node.fce == 4) res = vbicq_u32(ones, in1);
#endif
#if FUNCTIONS >= 6
        else if (node.fce == 5) res = vbicq_u32(ones, in2);
#endif
#if FUNCTIONS >= 7
        else if (node.fce == 6) res = vandq_u32(in1, vbicq_u32(ones, in2));
#endif
#if FUNCTIONS >= 8
        else if (node.fce == 7) res = veorq_u32(vandq_u32(in1, in2), ones);
#endif
#if FUNCTIONS >= 9
        else if (node.fce == 8) res = veorq_u32(vorrq_u32(in1, in2), ones);
#endif
#endif
        v_vystupy[node.out_idx] = res;
    }
    
    int total_correct = 0;
    for (i = 0; i < param_out; i++) {
#if defined(__AVX2__)
        SIMD_TYPE matched = _mm256_xor_si256(_mm256_xor_si256(v_vystupy[ac.out_indices[i]], p_svystup[i]), ones);
        matched = _mm256_and_si256(matched, valid_mask);
#else
        SIMD_TYPE matched = veorq_u32(veorq_u32(v_vystupy[ac.out_indices[i]], p_svystup[i]), ones);
        matched = vandq_u32(matched, valid_mask);
#endif
        total_correct += popcount_simd(matched);
    }
    
    return total_correct;
}

//-----------------------------------------------------------------------
//OHODNOCENI POPULACE
//=======================================================================
inline void ohodnoceni(SIMD_TYPE *vstup_komb, int minidx, int maxidx, int ignoreidx) {
    // NOTE(Sigull): When timed this is more than 97% (rest is probably precompute_active)
    int total_vars = param_in + param_out;

    for (int i = minidx; i < maxidx; i++) {
        if (i != ignoreidx) fitt[i] = 0;
    }

    for (int l = 0; l < param_fitev; l++) {
        // Pointer to the batch of SIMD_WIDTH * 32 test cases
        SIMD_TYPE* batch_ptr = vstup_komb + (l * total_vars);
        
        // Copy just the inputs to our 'vystupy' workbench
        memcpy(vystupy, batch_ptr, param_in * sizeof(SIMD_TYPE));
        
        // Grab the pre-calculated mask and outputs for this batch
        SIMD_TYPE current_mask = valid_masks[l];
        SIMD_TYPE* expected_outputs = batch_ptr + param_in;

        for (int i = minidx; i < maxidx; i++) {
            if (i == ignoreidx) continue;

            // if (fitt[i] + (param_fitev - l) * (SIMD_WIDTH * 32) * param_out < bestfit) continue; // Needs to be adjusted for blk
            if (active_popul[i].active_count > bestblk) continue;
            if (!(active_popul[i].active_count < bestblk) &&
                  (fitt[i] + (param_fitev - l) * (SIMD_WIDTH * 32) * param_out < bestfit)) continue;
            fitt[i] += fitness(i, expected_outputs, current_mask);
        }
    }
}

//-----------------------------------------------------------------------
//MUTACE
//=======================================================================
//p_chrom ukazatel na chromozom, jenz se ma zmutovat
//-----------------------------------------------------------------------
inline void mutace(chromozom p_chrom) {
    int rnd;
    int genu = (rand()%MUTACE_MAX) + 1;     //pocet genu, ktere se budou mutovat
    for (int j = 0; j < genu; j++) {
        int i = rand() % (outputidx + param_out); //vyber indexu v chromozomu pro mutaci
        int sloupec = (int) (i / sizesloupec);
        rnd = rand();
        if (i < outputidx) { //mutace bloku
           if ((i % 3) < 2) { //mutace vstupu
              p_chrom[i] = sloupce_val[sloupec]->hodnoty[(rnd % (sloupce_val[sloupec]->pocet))];
           } else { //mutace fce
              p_chrom[i] = rnd % FUNCTIONS;
           }
        } else { //mutace vystupu
           p_chrom[i] = rnd % maxidx_out;
        }
    }
}

void init_avx_data() {
    int total_vars = param_in + param_out;
    int blocks_32 = DATASIZE / total_vars;
    param_fitev = (blocks_32 + SIMD_WIDTH - 1) / SIMD_WIDTH;
    
#if defined(__AVX2__)
    tdata = (SIMD_TYPE*)_mm_malloc(param_fitev * total_vars * sizeof(SIMD_TYPE), 32);
    valid_masks = (SIMD_TYPE*)_mm_malloc(param_fitev * sizeof(SIMD_TYPE), 32);
#else
    tdata = (SIMD_TYPE*)aligned_alloc(32, param_fitev * total_vars * sizeof(SIMD_TYPE));
    valid_masks = (SIMD_TYPE*)aligned_alloc(32, param_fitev * sizeof(SIMD_TYPE));
#endif
    
    memset(tdata, 0, param_fitev * total_vars * sizeof(SIMD_TYPE));
    int* tdata_as_int = (int*)tdata;
    for (int b = 0; b < blocks_32; b++) {
        int avx_block = b / SIMD_WIDTH;
        int lane = b % SIMD_WIDTH;
        for (int v = 0; v < total_vars; v++) {
            int dest_idx = (avx_block * total_vars * SIMD_WIDTH) + (v * SIMD_WIDTH) + lane;
            int src_idx = (b * total_vars) + v;
            tdata_as_int[dest_idx] = tdata_int[src_idx];
        }
    }
    
    int leftover_blocks = blocks_32 % SIMD_WIDTH;
    for (int l = 0; l < param_fitev; l++) {
        if (l == param_fitev - 1 && leftover_blocks != 0) {
            memset(&valid_masks[l], 0, sizeof(SIMD_TYPE));
            memset(&valid_masks[l], 0xFF, leftover_blocks * sizeof(int));
        } else {
#if defined(__AVX2__)
            valid_masks[l] = _mm256_set1_epi32(-1);
#else
            valid_masks[l] = vdupq_n_u32((uint32_t)-1);
#endif
        }
    }
}

void shuffle_subarray(int* arr, int start, int end) {
    for (int i = end; i > start; i--) {
        int j = start + rand() % (i - start + 1);

        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
}

void vertical_shuffle(int popi) {
    chromozom p_chrom = (chromozom)populace[popi];

    int map[ARRSIZE + PARAM_IN];
    int inv_map[ARRSIZE + PARAM_IN];
    for (int i=0; i < ARRSIZE + param_in; i++) {
        map[i] = i;
        inv_map[i] = i;
    }

    for (int i=param_in; i < ARRSIZE + param_in; i+=param_n) {
        shuffle_subarray(inv_map, i, i+param_n-1);
    }
    for (int i=param_in; i < param_in + ARRSIZE; i++) {
        map[inv_map[i]] = i;
    }

    chromozom temp_chrom = new int [outputidx + param_out];
    copy_chromozome(p_chrom, temp_chrom);
    for (int i=0; i < ARRSIZE; i++) {
        int* gen_in = temp_chrom + i*3;
        int* gen_out = p_chrom + (map[i+param_in] - param_in)*3;
        memcpy(gen_out, gen_in, sizeof(int)*3);
        gen_out[0] = map[gen_out[0]];
        gen_out[1] = map[gen_out[1]];
    }

    for (int i=0; i < param_out; i++) {
        *(p_chrom + outputidx + i) = map[*(p_chrom + outputidx + i)];
    }
    delete[] temp_chrom;
}

void horizontal_migrate(int popi) {
    chromozom p_chrom = (chromozom)populace[popi];
    const ActiveChrom& ac = active_popul[popi];

    // Build active set for quick lookup
    bool is_active[ARRSIZE + PARAM_IN] = {};
    for (const auto& node : ac.nodes)
        is_active[node.out_idx] = true;

    for (const auto& node : ac.nodes) {
        int out_a  = node.out_idx;
        int flat_a = out_a - param_in;        // flat index of node A
        int col_a  = flat_a / param_n;

        if (col_a == 0) continue;             // nothing left to migrate into
        int col_b = col_a - 1;

        // Check A's inputs are reachable from col_b
        int min_reach = param_in + std::max(0, col_b - l_back) * param_n;
        int max_reach = param_in + col_b * param_n;  // exclusive

        auto reachable = [&](int in) {
            return in < param_in || (in >= min_reach && in < max_reach);
        };
        if (!reachable(node.in1) || !reachable(node.in2)) continue;

        // Find an inactive slot at col_b to swap with
        int flat_b = -1;
        for (int r = 0; r < param_n; r++) {
            int candidate = param_in + col_b * param_n + r;
            if (!is_active[candidate]) {
                flat_b = col_b * param_n + r;
                break;
            }
        }
        if (flat_b == -1) continue;

        int out_b = param_in + flat_b;

        // Check no downstream node referencing out_a would violate l_back
        // after A moves to col_b. Fails only when referencer is at col_a + l_back exactly.
        bool safe = true;
        for (int i = 0; i < outputidx; i += 3) {
            if (p_chrom[i] == out_a || p_chrom[i+1] == out_a) {
                int ref_col = (i / 3) / param_n;
                if (ref_col - col_b > l_back) { safe = false; break; }
            }
        }
        if (!safe) continue;

        int* gene_a = p_chrom + flat_a * 3;
        int* gene_b = p_chrom + flat_b * 3;
        int tmp[3];
        memcpy(tmp,    gene_a, 3 * sizeof(int));
        memcpy(gene_a, gene_b, 3 * sizeof(int));
        memcpy(gene_b, tmp,    3 * sizeof(int));

        // Remap all out_a to out_b throughout the chromosome.
        for (int i = 0; i < outputidx; i += 3) {
            for (int k = 0; k < 2; k++) {
                if      (p_chrom[i+k] == out_a) p_chrom[i+k] = out_b;
                else if (p_chrom[i+k] == out_b) p_chrom[i+k] = out_a;
            }
        }
        for (int i = 0; i < param_out; i++) {
            int& o = p_chrom[outputidx + i];
            if      (o == out_a) o = out_b;
            else if (o == out_b) o = out_a;
        }

        // One migration per call
        break;
    }
}

//-----------------------------------------------------------------------
// MAIN
//-----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    using namespace std;

    FILE *xlsfil, *colcount;
    string logfname, logfname2;
    int rnd, fitn, blk;
    int *vstup_komb; //ukazatel na vstupni data
    bool log;
    int run_succ = 0;
    int i;
    int parentidx;
    
    logfname = "log";
    if ((argc == 2) && (argv[1] != "")) 
       logfname = string(argv[1]);
    
    vystupy = (int*)
#if defined(__AVX2__)
    _mm_malloc((maxidx_out + param_out) * sizeof(SIMD_TYPE), 32);
#else
    aligned_alloc(32, (maxidx_out + param_out) * sizeof(SIMD_TYPE));
#endif
    pouzite = new int [maxidx_out];

    init_data(tdata_int);
    init_avx_data();

    int total_vars = param_in + param_out;
    int blocks_32 = DATASIZE / total_vars; // Exactly how many 32-bit blocks we have (e.g. 5)
    param_fitev = CEIL_DIV(blocks_32, SIMD_WIDTH);

    maxfitness = blocks_32 * 32 * param_out; 
    fitnessepsilon = (int)(maxfitness * fitepsilon);
    assert(maxfitness + maxblkfitness > 0); //Sanity check

    unsigned seed = 1776116149;
    srand(seed); //inicializace pseudonahodneho generatoru
    printf("Seed of run/s is %d\n", seed);

    /**
    // Not cache local
    for (int i=0; i < param_populace; i++) //alokace pameti pro chromozomy populace
        populace[i] = new chromozom [outputidx + param_out];
    */

    // Doesn't help much in the end.
    // Cache local equivalent. Could fail if too big.
    // chromozom* populace_arena = nullptr;
    size_t n_jeden_chromozon = outputidx + param_out;
    try {
        int* raw_arena = new int[param_populace * n_jeden_chromozon];
        populace_arena = (chromozom*)raw_arena;
        for (int i = 0; i < param_populace; i++) {
            populace[i] = (chromozom*)(raw_arena + (n_jeden_chromozon * i));
        }
    }
    catch (const std::bad_alloc& e) {
        printf("Fallback to individual chromozome allocation!\n");
        for (int i=0; i < param_populace; i++) 
        populace[i] = new chromozom [outputidx + param_out];
    }

    //---------------------------------------------------------------------------
    // Vytvoreni LOOKUP tabulky pro rychle zjisteni poctu nenulovych bitu v bytu
    //---------------------------------------------------------------------------
    for (int i = 0; i < LOOKUPTABSIZE; i++) {
        int poc1 = 0;
        int zi = ~i;
        for (int j=0; j < 8; j++) {
            poc1 += (zi & 1);
            zi = zi >> 1;
        }
        lookupbit_tab[i] = poc1;
    }

    //-----------------------------------------------------------------------
    //Priprava pole moznych hodnot vstupu pro sloupec podle l-back a ostatnich parametru
    //-----------------------------------------------------------------------
    sloupce_val = new sl_rndval *[param_m]; // 1. Array of pointers
    
    for (int i=0; i < param_m; i++) {
        sloupce_val[i] = new sl_rndval; // 2. Allocate the struct itself

        int minidx = param_n*(i-l_back) + param_in;
        if (minidx < param_in) minidx = param_in; 
        int maxidx = i*param_n + param_in;

        sloupce_val[i]->pocet = param_in + maxidx - minidx;
        
        // 3. Allocate the integer array inside the struct
        sloupce_val[i]->hodnoty = new int [sloupce_val[i]->pocet];

        int j=0;
        for (int k=0; k < param_in; k++,j++) 
            sloupce_val[i]->hodnoty[j] = k;
        for (int k=minidx; k < maxidx; k++,j++) 
            sloupce_val[i]->hodnoty[j] = k;
    }

    //-----------------------------------------------------------------------
    printf("LogName: %s  l-back: %d  popsize:%d\n", logfname.c_str(), l_back, param_populace);
    //-----------------------------------------------------------------------

    for (int run=0; run < PARAM_RUNS; run++) {
        time_t t;
        struct tm *tl;
        char fn[100];
        char cfn[100];
    
        t = time(NULL);
        tl = localtime(&t);
    
        snprintf(fn, sizeof(fn)-1, "_%d", run);
        logfname2 = logfname + string(fn);
        strcpy(fn, logfname2.c_str()); strcat(fn,".xls");
        xlsfil = fopen(fn,"wb");
        strcpy(cfn, logfname2.c_str()); strcat(cfn,"_col.txt");
        colcount = fopen(cfn,"w");
        if (!xlsfil || !colcount) {
           printf("Can't create file %s!\n",fn);
           return -1;
        }
        fprintf(colcount, "Seed: %d\n", seed);

        //Hlavicka do log. souboru
        fprintf(xlsfil, "Generation\tBestfitness\tPop. fitness\t#Blocks\t\t");
        for (int i=0; i < param_populace;  i++) fprintf(xlsfil,"chrom #%d\t",i);
        fprintf(xlsfil, "\n");
    
        printf("----------------------------------------------------------------\n");
        printf("Run: %d \t\t %s", run, asctime(tl));
        printf("----------------------------------------------------------------\n");
    
        //-----------------------------------------------------------------------
        //Vytvoreni pocatecni populace
        //-----------------------------------------------------------------------
        chromozom p_chrom;
        int sloupec;
        for (int i=0; i < param_populace; i++) {
            p_chrom = (chromozom) populace[i];
            for (int j=0; j < param_m*param_n; j++) {
                sloupec = (int)(j / param_n);
                // vstup 1
                *p_chrom++ = sloupce_val[sloupec]->hodnoty[(rand() % (sloupce_val[sloupec]->pocet))];
                // vstup 2
                *p_chrom++ = sloupce_val[sloupec]->hodnoty[(rand() % (sloupce_val[sloupec]->pocet))];
                // funkce
                rnd = rand() % FUNCTIONS;
                *p_chrom++ = rnd;
            }
            for (int j=outputidx; j < outputidx+param_out; j++)  //napojeni vystupu
                *p_chrom++ = rand() % maxidx_out;

            precompute_active(i);
        }

        // If chrin (CHROMOZOME_LOAD) defined, put it into the first population.
        // Will probably be chosen for the next.
        chromozom temp_chrom = new int [outputidx + param_out];
        if (load_chrom(chrin, sizeof(chrin), temp_chrom)) {
            copy_chromozome(temp_chrom, populace[0]);
            precompute_active(0);
        } else {
            printf("Error in CHROMOZOME_LOAD. Not matching params or malformed.\n");
        }
        delete[] temp_chrom;

        //-----------------------------------------------------------------------
        //Ohodnoceni pocatecni populace
        //-----------------------------------------------------------------------
        bestfit = 0; bestfit_idx = -1, bestblk = ARRSIZE, last_improvement = 0;
        ohodnoceni(tdata /*vektor ocekavanych dat*/, 0, param_populace, -1);
        for (int i=0; i < param_populace; i++) { //nalezeni nejlepsiho jedince
            if (fitt[i] > bestfit) {
               bestfit = fitt[i];
               bestfit_idx = i;
            }
        }
    
        //bestfit_idx ukazuje na nejlepsi reseni v ramci pole jedincu "populace"
        //bestfit obsahuje fitness hodnotu prvku s indexem bestfit_idx

        if (bestfit_idx == -1) 
           return 0;
    
        //-----------------------------------------------------------------------
        // EVOLUCE
        //-----------------------------------------------------------------------
        param_generaci = 0;
        maxfitpop = 0;
        while (param_generaci++ < PARAM_GENERATIONS) {
            //-----------------------------------------------------------------------
            //Periodicky vypis chromozomu populace
            //-----------------------------------------------------------------------
            #ifdef PERIODIC_LOG
            if (param_generaci % PERIODICLOGG == 0) {
               printf("Generation: %d\n",param_generaci);
               for(int j=0; j<param_populace; j++) {
                  printf("{%d, %d}",fitt[j],uzitobloku((int *)populace[j]));
                  print_chrom(stdout,(chromozom)populace[j]);
               }
            }
            #endif

            //-----------------------------------------------------------------------
            //mutace nejlepsiho jedince populace (na param_populace mutantu)
            //-----------------------------------------------------------------------
            for (int i=0, midx = 0; i < param_populace;  i++, midx++) {
                if (bestfit_idx == i) continue;

                p_chrom = (int *) copy_chromozome(populace[bestfit_idx],populace[midx]);
                mutace(p_chrom);
                precompute_active(i);
            }

            if (param_generaci - last_improvement >= N_SHUFFLE) {
                printf("Generation:%d shuffled\n", param_generaci);
                last_improvement = param_generaci;
                for (int i=0; i < POPULACE_MAX; i++) {
                    vertical_shuffle(i);
                    for (int j=0; j < SHUFFLE_COUNT; j++) {
                        horizontal_migrate(i);
                    }
                    precompute_active(i);
                }
                log_active_nodes_by_column(colcount, bestfit_idx);
                fflush(colcount);
            }

            //-----------------------------------------------------------------------
            //ohodnoceni populace
            //-----------------------------------------------------------------------
            ohodnoceni(tdata, 0, param_populace, bestfit_idx);
            parentidx = bestfit_idx;
            fitpop = 0;
            log = false;
            for (int i=0; i < param_populace; i++) { 
                fitpop += fitt[i];
                
                if (i == parentidx) continue; //preskocime rodice

                if (fitt[i] >= fitnessepsilon) {
                   //optimalizace na poc. bloku obvodu

                    blk = uzitobloku((chromozom) populace[i]);
                    if (blk <= bestblk) {

                        if (blk < bestblk) {
                            printf("Generation:%d\t\tbestblk b:%d\n",param_generaci,blk);
                            log = true;
                            last_improvement = param_generaci;
                        }
                        
                        if (blk < bestblk || bestfit <= fitt[i]) {
                            if (blk < bestblk || bestfit < fitt[i]) {
                                log_active_nodes_by_column(colcount, bestfit_idx);
                                fflush(colcount);
                            }
                            bestfit_idx = i;
                            bestfit = fitt[i];
                            bestblk = blk;
                        }
                    }
                } else if (fitt[i] >= bestfit) {
                   //nalezen lepsi nebo stejne dobry jedinec jako byl jeho rodic

                   if (fitt[i] > bestfit) {
                      printf("Generation:%d\t\tFittness: %d/%d\n",param_generaci,fitt[i],maxfitness);
                      log = true;
                   }

                   bestfit_idx = i;
                   bestfit = fitt[i];
                   bestblk = ARRSIZE;
                //    log_active_nodes_by_column(stdout, bestfit_idx);
                }
            }
    
            //-----------------------------------------------------------------------
            // Vypis fitness populace do xls souboru pri zmene fitness populace/poctu bloku
            //-----------------------------------------------------------------------
            if (log) {
               print_xls(xlsfil);

               maxfitpop = fitpop;
               log = false;
            }
        }
        //-----------------------------------------------------------------------
        // Konec evoluce
        //-----------------------------------------------------------------------
        print_xls(xlsfil);
        fclose(xlsfil);
        printf("Best chromosome fitness: %d/%d\n",bestfit,maxfitness);
        printf("Best chromosome blk: %d/%d\n",bestblk,ARRSIZE);
        log_active_nodes_by_column(stdout, bestfit_idx);
        printf("Best chromosome: ");
        print_chrom(stdout, (chromozom)populace[bestfit_idx]);
    
        if (bestfit == maxfitness) {
            strcpy(fn, logfname2.c_str()); strcat(fn,".chr");
            FILE *chrfil = fopen(fn,"wb");
            fprintf(chrfil, POPIS);
            print_chrom(chrfil, (chromozom)populace[bestfit_idx]);
            fclose(chrfil);
        }

        if (bestfit == maxfitness) 
           run_succ++; 
    } //runs

    /**
    //Not cache local
    for (int i=param_populace-1; i >= 0; i--)
        delete[] populace[i];
    */

    printf("Successful runs: %d/%d (%5.1f%%)\n",run_succ, PARAM_RUNS, 100*run_succ/(float)PARAM_RUNS);

    //Cache local but doesnt help in the end.
    if (populace_arena) {
        delete[] (int*)populace_arena;
        
    } else {
        for (int i = param_populace - 1; i >= 0; i--)
            delete[] populace[i];
    }
    return 0;
}

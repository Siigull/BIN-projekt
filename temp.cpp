#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <immintrin.h>
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
 
int param_fitev;  //pocet pruchodu pro ohodnoceni jednoho chromozomu, vznikne jako (pocet vstupnich dat/(pocet vstupu+pocet vystupu))
 
int param_generaci; //pocet kroku evoluce
int tdata_int[DATASIZE];
 
#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define DATASIZE_256 CEIL_DIV(DATASIZE, 8)
__m256i* tdata;
__m256i* valid_masks;
 
int sizesloupec = param_n*(block_in+1); //pocet polozek ktery zabira sloupec v chromozomu
int outputidx   = param_m*sizesloupec; //index v poli chromozomu, kde zacinaji vystupy
int maxidx_out  = param_n*param_m + param_in; //max. index pouzitelny jako vstup  pro vystupy
int maxfitness  = 0; //max. hodnota fitness
 
int fitpop, maxfitpop; //fitness populace
 
double fitepsilon = 0.95;
int fitnessepsilon = 0;
int maxblkfitness = PARAM_M * PARAM_N;
 
typedef struct {
    int pocet;
    int *hodnoty;
} sl_rndval;
 
sl_rndval **sloupce_val;
#define ARRSIZE PARAM_M*PARAM_N
 
#define LOOKUPTABSIZE 256
unsigned char lookupbit_tab[LOOKUPTABSIZE];
 
#define copy_chromozome(from,to) (chromozom *) memcpy(to, from, (outputidx + param_out)*sizeof(int));
 
#define FITNESS_CALLCNT (POPULACE_MAX + PARAM_GENERATIONS*POPULACE_MAX)
 
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
 
int uzitobloku(chromozom p_chrom) {
    int i,j, in,idx, poc = 0;
    int *p_pom;
    memset(pouzite, 0, maxidx_out*sizeof(int));
 
    p_pom = p_chrom + outputidx;
    for (i=0; i < param_out; i++) {
        in = *p_pom++;
        pouzite[in] = 1;
    }
 
    p_pom = p_chrom + outputidx - 1;
    idx = maxidx_out-1;
    for (i=param_m; i > 0; i--) {
        for (j=param_n; j > 0; j--,idx--) {
            p_pom--;
            if (pouzite[idx] == 1) {
               in = *p_pom--;
               pouzite[in] = 1;
               in = *p_pom--;
               pouzite[in] = 1;
               poc++;
            } else {
               p_pom -= block_in;
            }
        }
    }
 
    return poc;
}
 
inline int popcount_256_avx2(__m256i v) {
    const __m256i lookup = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4
    );
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
 
    __m256i lo = _mm256_and_si256(v, low_mask);
    __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
 
    __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
    __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
 
    __m256i total = _mm256_add_epi8(popcnt1, popcnt2);
    total = _mm256_sad_epu8(total, _mm256_setzero_si256());
 
    return (int)(_mm256_extract_epi64(total, 0) +
                 _mm256_extract_epi64(total, 1) +
                 _mm256_extract_epi64(total, 2) +
                 _mm256_extract_epi64(total, 3));
}
 
struct ActiveChrom {
    struct Node { int in1, in2, fce, out_idx; };
    std::vector<Node> nodes;  // Only active nodes, topologically sorted
    int out_indices[PARAM_OUT];
    int active_count;
};
 
ActiveChrom active_popul[POPULACE_MAX];
 
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
    int idx_loop = maxidx_out - 1;
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
                p_pom -= (block_in + 1);
            }
        }
    }
 
    std::reverse(rev.begin(), rev.end());
    ac.nodes = std::move(rev);
    ac.active_count = (int)ac.nodes.size();
}
 
//-----------------------------------------------------------------------
// FITNESS
// Evaluates one individual against one AVX batch of test cases.
// Returns number of correct output bits.
//-----------------------------------------------------------------------
inline int fitness(int pop_idx, const __m256i* __restrict p_svystup, __m256i valid_mask) {
    const ActiveChrom& ac = active_popul[pop_idx];
    __m256i* v_vystupy = (__m256i*)vystupy;
 
    static const __m256i ones = _mm256_set1_epi32(-1);
 
    for (const auto& node : ac.nodes) {
        __m256i in1 = v_vystupy[node.in1];
        __m256i in2 = v_vystupy[node.in2];
        __m256i res;
 
        if      (node.fce == 0) res = in1;
        else if (node.fce == 1) res = _mm256_and_si256(in1, in2);
        else if (node.fce == 2) res = _mm256_or_si256(in1, in2);
        else if (node.fce == 3) res = _mm256_xor_si256(in1, in2);
        else if (node.fce == 4) res = _mm256_andnot_si256(in1, ones);                       // NOT in1
        else if (node.fce == 7) res = _mm256_xor_si256(_mm256_and_si256(in1, in2), ones);   // NAND
        else if (node.fce == 8) res = _mm256_xor_si256(_mm256_or_si256(in1, in2),  ones);   // NOR
        else { assert(false && "Unknown function"); __builtin_unreachable(); }
 
        v_vystupy[node.out_idx] = res;
    }
 
    int total_correct = 0;
    for (int i = 0; i < param_out; i++) {
        __m256i matched = _mm256_xor_si256(
            _mm256_xor_si256(v_vystupy[ac.out_indices[i]], p_svystup[i]),
            ones);
        matched = _mm256_and_si256(matched, valid_mask);
        total_correct += popcount_256_avx2(matched);
    }
 
    return total_correct;
}
 
//-----------------------------------------------------------------------
// OHODNOCENI POPULACE
//
// Optimizations applied:
//   1. Full batches (all but last) use a hardcoded all-ones mask, avoiding
//      a redundant _mm256_and_si256 per output per individual per batch.
//   2. Early termination: once an individual's maximum achievable score
//      can no longer reach bestfit, skip remaining batches for that individual.
//      The threshold uses strict less-than so we keep individuals that
//      could at least tie the current best (selection accepts >= bestfit).
//-----------------------------------------------------------------------
inline void ohodnoceni(__m256i *vstup_komb, int minidx, int maxidx, int ignoreidx) {
    const int total_vars    = param_in + param_out;
    // Maximum bits an individual can gain from one full (256-case) batch
    const int max_per_batch = 256 * param_out;
 
    for (int i = minidx; i < maxidx; i++)
        if (i != ignoreidx) fitt[i] = 0;
 
    static const __m256i full_mask = _mm256_set1_epi32(-1);
 
    // ── Full batches (no padding mask needed) ──────────────────────────
    const int full_batches = param_fitev - 1;
    for (int l = 0; l < full_batches; l++) {
        __m256i* batch_ptr       = vstup_komb + (l * total_vars);
        __m256i* expected_outputs = batch_ptr + param_in;
 
        memcpy(vystupy, batch_ptr, param_in * sizeof(__m256i));
 
        const int batches_left = param_fitev - l; // includes current batch
        for (int i = minidx; i < maxidx; i++) {
            if (i == ignoreidx) continue;
            // Early exit: even a perfect score from here can't reach bestfit
            if (fitt[i] + batches_left * max_per_batch < bestfit) continue;
 
            fitt[i] += fitness(i, expected_outputs, full_mask);
        }
    }
 
    // ── Last batch (apply padding validity mask) ───────────────────────
    {
        const int l              = param_fitev - 1;
        __m256i* batch_ptr       = vstup_komb + (l * total_vars);
        __m256i* expected_outputs = batch_ptr + param_in;
 
        memcpy(vystupy, batch_ptr, param_in * sizeof(__m256i));
 
        for (int i = minidx; i < maxidx; i++) {
            if (i == ignoreidx) continue;
            if (fitt[i] + max_per_batch < bestfit) continue; // 1 batch left
 
            fitt[i] += fitness(i, expected_outputs, valid_masks[l]);
        }
    }
}
 
//-----------------------------------------------------------------------
//MUTACE
//-----------------------------------------------------------------------
inline void mutace(chromozom p_chrom) {
    int rnd;
    int genu = (rand()%MUTACE_MAX) + 1;
    for (int j = 0; j < genu; j++) {
        int i = rand() % (outputidx + param_out);
        int sloupec = (int) (i / sizesloupec);
        rnd = rand();
        if (i < outputidx) {
           if ((i % 3) < 2) {
              p_chrom[i] = sloupce_val[sloupec]->hodnoty[(rnd % (sloupce_val[sloupec]->pocet))];
           } else {
              p_chrom[i] = rnd % FUNCTIONS;
           }
        } else {
           p_chrom[i] = rnd % maxidx_out;
        }
    }
}
 
void init_avx_data() {
    int total_vars = param_in + param_out;
    
    int blocks_32 = DATASIZE / total_vars; 
    param_fitev = (blocks_32 + 7) / 8;
    
    if (param_fitev == 0) param_fitev = 1;
 
    maxfitness = blocks_32 * 32 * param_out; 
    fitnessepsilon = (int)(maxfitness * fitepsilon);
    assert(maxfitness + maxblkfitness > 0); 
 
    tdata = (__m256i*)_mm_malloc(param_fitev * total_vars * sizeof(__m256i), 32);
    valid_masks = (__m256i*)_mm_malloc(param_fitev * sizeof(__m256i), 32);
    
    memset(tdata, 0, param_fitev * total_vars * sizeof(__m256i));
 
    int* tdata_as_int = (int*)tdata;
    for (int b = 0; b < blocks_32; b++) {
        int avx_block = b / 8;
        int lane = b % 8;
        for (int v = 0; v < total_vars; v++) {
            int dest_idx = (avx_block * total_vars * 8) + (v * 8) + lane;
            int src_idx = (b * total_vars) + v;
            tdata_as_int[dest_idx] = tdata_int[src_idx];
        }
    }
 
    int leftover_blocks = blocks_32 % 8;
    for (int l = 0; l < param_fitev; l++) {
        if (l == param_fitev - 1 && leftover_blocks != 0) {
            memset(&valid_masks[l], 0,    sizeof(__m256i));
            memset(&valid_masks[l], 0xFF, leftover_blocks * sizeof(int));
        } else {
            valid_masks[l] = _mm256_set1_epi32(-1);
        }
    }
}
 
//-----------------------------------------------------------------------
// MAIN
//-----------------------------------------------------------------------
int main(int argc, char* argv[])
{
    using namespace std;
 
    FILE *xlsfil;
    string logfname, logfname2;
    int rnd, fitn, blk;
    int *vstup_komb;
    bool log;
    int run_succ = 0;
    int i;
    int parentidx;
    
    logfname = "log";
    if ((argc == 2) && (argv[1] != "")) 
       logfname = string(argv[1]);
    
    vystupy = (int*)_mm_malloc((maxidx_out + param_out) * sizeof(__m256i), 32);
    pouzite = new int [maxidx_out];
 
    init_data(tdata_int);
    init_avx_data();
 
    int total_vars = param_in + param_out;
    int blocks_32 = DATASIZE / total_vars;
    param_fitev = CEIL_DIV(blocks_32, 8);
 
    maxfitness = blocks_32 * 32 * param_out; 
    fitnessepsilon = (int)(maxfitness * fitepsilon);
    assert(maxfitness + maxblkfitness > 0);
 
    srand(41);
    
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
 
    for (int i = 0; i < LOOKUPTABSIZE; i++) {
        int poc1 = 0;
        int zi = ~i;
        for (int j=0; j < 8; j++) {
            poc1 += (zi & 1);
            zi = zi >> 1;
        }
        lookupbit_tab[i] = poc1;
    }
 
    sloupce_val = new sl_rndval *[param_m];
    
    for (int i=0; i < param_m; i++) {
        sloupce_val[i] = new sl_rndval;
 
        int minidx = param_n*(i-l_back) + param_in;
        if (minidx < param_in) minidx = param_in; 
        int maxidx = i*param_n + param_in;
 
        sloupce_val[i]->pocet = param_in + maxidx - minidx;
        sloupce_val[i]->hodnoty = new int [sloupce_val[i]->pocet];
 
        int j=0;
        for (int k=0; k < param_in; k++,j++) 
            sloupce_val[i]->hodnoty[j] = k;
        for (int k=minidx; k < maxidx; k++,j++) 
            sloupce_val[i]->hodnoty[j] = k;
    }
 
    printf("LogName: %s  l-back: %d  popsize:%d\n", logfname.c_str(), l_back, param_populace);
 
    for (int run=0; run < PARAM_RUNS; run++) {
        time_t t;
        struct tm *tl;
        char fn[100];
    
        t = time(NULL);
        tl = localtime(&t);
    
        sprintf(fn, "_%d", run);
        logfname2 = logfname + string(fn);
        strcpy(fn, logfname2.c_str()); strcat(fn,".xls");
        xlsfil = fopen(fn,"wb");
        if (!xlsfil) {
           printf("Can't create file %s!\n",fn);
           return -1;
        }
        fprintf(xlsfil, "Generation\tBestfitness\tPop. fitness\t#Blocks\t\t");
        for (int i=0; i < param_populace;  i++) fprintf(xlsfil,"chrom #%d\t",i);
        fprintf(xlsfil, "\n");
    
        printf("----------------------------------------------------------------\n");
        printf("Run: %d \t\t %s", run, asctime(tl));
        printf("----------------------------------------------------------------\n");
    
        chromozom p_chrom;
        int sloupec;
        for (int i=0; i < param_populace; i++) {
            p_chrom = (chromozom) populace[i];
            for (int j=0; j < param_m*param_n; j++) {
                sloupec = (int)(j / param_n);
                *p_chrom++ = sloupce_val[sloupec]->hodnoty[(rand() % (sloupce_val[sloupec]->pocet))];
                *p_chrom++ = sloupce_val[sloupec]->hodnoty[(rand() % (sloupce_val[sloupec]->pocet))];
                rnd = rand() % FUNCTIONS;
                *p_chrom++ = rnd;
            }
            for (int j=outputidx; j < outputidx+param_out; j++)
                *p_chrom++ = rand() % maxidx_out;
 
            precompute_active(i);
        }
 
        bestfit = 0; bestfit_idx = -1;
        ohodnoceni(tdata, 0, param_populace, -1);
        for (int i=0; i < param_populace; i++) {
            if (fitt[i] > bestfit) {
               bestfit = fitt[i];
               bestfit_idx = i;
            }
        }
    
        if (bestfit_idx == -1) 
           return 0;
    
        param_generaci = 0;
        maxfitpop = 0;
        while (param_generaci++ < PARAM_GENERATIONS) {
            #ifdef PERIODIC_LOG
            if (param_generaci % PERIODICLOGG == 0) {
               printf("Generation: %d\n",param_generaci);
               for(int j=0; j<param_populace; j++) {
                  printf("{%d, %d}",fitt[j],uzitobloku((int *)populace[j]));
                  print_chrom(stdout,(chromozom)populace[j]);
               }
            }
            #endif
 
            for (int i=0, midx = 0; i < param_populace;  i++, midx++) {
                if (bestfit_idx == i) continue;
 
                p_chrom = (int *) copy_chromozome(populace[bestfit_idx],populace[midx]);
                mutace(p_chrom);
                precompute_active(i);
            }
 
            ohodnoceni(tdata, 0, param_populace, bestfit_idx);
            parentidx = bestfit_idx;
            fitpop = 0;
            log = false;
            for (int i=0; i < param_populace; i++) { 
                fitpop += fitt[i];
                
                if (i == parentidx) continue;
 
                if (fitt[i] == maxfitness) {
                   blk = uzitobloku((chromozom) populace[i]);
                   if (blk <= bestblk) {
                      if (blk < bestblk) {
                         printf("Generation:%d\t\tbestblk b:%d\n",param_generaci,blk);
                         log = true;
                      }
                      bestfit_idx = i;
                      bestfit = fitt[i];
                      bestblk = blk;
                   }
                } else if (fitt[i] >= bestfit) {
                   if (fitt[i] > bestfit) {
                      printf("Generation:%d\t\tFittness: %d/%d\n",param_generaci,fitt[i],maxfitness);
                      log = true;
                   }
                   bestfit_idx = i;
                   bestfit = fitt[i];
                   bestblk = ARRSIZE;
                }
            }
    
            if ((fitpop > maxfitpop) || (log)) {
               print_xls(xlsfil);
               maxfitpop = fitpop;
               log = false;
            }
        }
 
        print_xls(xlsfil);
        fclose(xlsfil);
        printf("Best chromosome fitness: %d/%d\n",bestfit,maxfitness);
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
    }
 
    printf("Successful runs: %d/%d (%5.1f%%)\n",run_succ, PARAM_RUNS, 100*run_succ/(float)PARAM_RUNS);
 
    if (populace_arena) {
        delete[] (int*)populace_arena;
    } else {
        for (int i = param_populace - 1; i >= 0; i--)
            delete[] populace[i];
    }
    return 0;
}
 
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include <immintrin.h>
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

//Rozšíření pro 
double fitepsilon = 0.95; //Splňuje cíl operace v epsilon zlomku populace = uspech
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

    return _mm256_extract_epi64(total, 0) + 
           _mm256_extract_epi64(total, 1) + 
           _mm256_extract_epi64(total, 2) + 
           _mm256_extract_epi64(total, 3);
}

inline int fitness(chromozom p_chrom, 
                   const __m256i* __restrict p_svystup, 
                   __m256i valid_mask) {
    int i, j;
    // We assume 'vystupy' is now an array of __m256i
    __m256i *v_vystupy = (__m256i*)vystupy;
    __m256i *p_v_vystup = v_vystupy + param_in;

    for (i = 0; i < param_m; i++) {
        for (j = 0; j < param_n; j++) {
            // Fetch inputs (Still 32-bit indices from chromosome)
            __m256i in1 = v_vystupy[*p_chrom++];
            __m256i in2 = v_vystupy[*p_chrom++];
            int fce = *p_chrom++;

            __m256i res;

            // NOTE(Sigull): Tried precomputing all values into an array -> was slower than ifs
            // NOTE(Sigull): Array function lookup table was also slower.
            if      (fce == 0) res = in1;
            else if (fce == 1) res = _mm256_and_si256(in1, in2);
            else if (fce == 2) res = _mm256_or_si256(in1, in2);
            else if (fce == 3) res = _mm256_xor_si256(in1, in2);
            else if (fce == 4) res = _mm256_andnot_si256(in1, _mm256_set1_epi32(-1)); // NOT in1
            else if (fce == 7) res = _mm256_xor_si256(_mm256_and_si256(in1, in2), _mm256_set1_epi32(-1)); // NAND
            else if (fce == 8) res = _mm256_xor_si256(_mm256_or_si256(in1, in2), _mm256_set1_epi32(-1));  // NOR
            else assert(false && "Should never get here.");

            *p_v_vystup++ = res;
        }
    }

    int total_correct = 0;
    for (i = 0; i < param_out; i++) {
        __m256i actual = v_vystupy[*p_chrom++];
        __m256i expected = p_svystup[i];

        __m256i matched = _mm256_xor_si256(_mm256_xor_si256(actual, expected), _mm256_set1_epi32(-1));
        matched = _mm256_and_si256(matched, valid_mask);
        total_correct += popcount_256_avx2(matched);
    }
    
    return total_correct;
}

//-----------------------------------------------------------------------
//OHODNOCENI POPULACE
//=======================================================================
inline void ohodnoceni(__m256i *vstup_komb, int minidx, int maxidx, int ignoreidx) {
    // NOTE(Sigull): When timed this is more than 99%
    int total_vars = param_in + param_out;

    for (int i = minidx; i < maxidx; i++) {
        if (i != ignoreidx) fitt[i] = 0;
    }

    for (int l = 0; l < param_fitev; l++) {
        // Pointer to the batch of 256 test cases
        __m256i* batch_ptr = vstup_komb + (l * total_vars);
        
        // Copy just the inputs to our 'vystupy' workbench
        memcpy(vystupy, batch_ptr, param_in * sizeof(__m256i));
        
        // Grab the pre-calculated mask and outputs for this batch
        __m256i current_mask = valid_masks[l];
        __m256i* expected_outputs = batch_ptr + param_in;

        for (int i = minidx; i < maxidx; i++) {
            if (i == ignoreidx) continue;

            fitt[i] += fitness((chromozom)populace[i], expected_outputs, current_mask);
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

    // 4. Pre-calculate the padding masks for every batch
    int leftover_blocks = blocks_32 % 8;
    for (int l = 0; l < param_fitev; l++) {
        if (l == param_fitev - 1 && leftover_blocks != 0) {
            memset(&valid_masks[l], 0, sizeof(__m256i));
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
    int *vstup_komb; //ukazatel na vstupni data
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
    int blocks_32 = DATASIZE / total_vars; // Exactly how many 32-bit blocks we have (e.g. 5)
    param_fitev = CEIL_DIV(blocks_32, 8);  // How many 256-bit batches we need

    maxfitness = blocks_32 * 32 * param_out; 
    fitnessepsilon = (int)(maxfitness * fitepsilon);
    assert(maxfitness + maxblkfitness > 0); //Sanity check

    srand(42); //inicializace pseudonahodneho generatoru
    
    /**
    //Not cache local
    for (int i=0; i < param_populace; i++) //alokace pameti pro chromozomy populace
        populace[i] = new chromozom [outputidx + param_out];
    */

    // Doesn't help in the end.
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
        }

        //-----------------------------------------------------------------------
        //Ohodnoceni pocatecni populace
        //-----------------------------------------------------------------------
        bestfit = 0; bestfit_idx = -1;
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

                if (fitt[i] == maxfitness) {
                   //optimalizace na poc. bloku obvodu

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
                   //nalezen lepsi nebo stejne dobry jedinec jako byl jeho rodic

                   if (fitt[i] > bestfit) {
                      printf("Generation:%d\t\tFittness: %d/%d\n",param_generaci,fitt[i],maxfitness);
                      log = true;
                   }

                   bestfit_idx = i;
                   bestfit = fitt[i];
                   bestblk = ARRSIZE;
                }
            }
    
            //-----------------------------------------------------------------------
            // Vypis fitness populace do xls souboru pri zmene fitness populace/poctu bloku
            //-----------------------------------------------------------------------
            if ((fitpop > maxfitpop) || (log)) {
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

#include <assert.h>
#include <stdio.h>
#include <stdint.h>

void print_bits(int8_t num, int nbits) {
    for (int i=0; i < nbits; i++) {
      int dig = (((int)num) >> (nbits - i - 1)) & 1;
      printf("%d", dig);
    }
}

int main() {
  int8_t result = 0;
  for (int8_t i = 0;; i++){
    for (int8_t j = 0;; j++){
      result = i + j;

      print_bits(i, 8);
      print_bits(j, 8);
      printf(" : ");
      print_bits(result, 8);
      printf("\n");
      if(j==-1) break;
    }
    if(i==-1) break;
  }
  return 0;
}

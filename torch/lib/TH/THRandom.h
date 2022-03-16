#ifndef TH_RANDOM_INC
#define TH_RANDOM_INC

#include "THGeneral.h"

#define _MERSENNE_STATE_N 624
#define _MERSENNE_STATE_M 397
/* A THGenerator contains all the state required for a single random number stream */
typedef struct THGenerator {
  /* The initial seed. */
  unsigned long the_initial_seed;
  int left;  /* = 1; */
  int seeded; /* = 0; */
  unsigned long next;
  unsigned long state[_MERSENNE_STATE_N]; /* the array for the state vector  */
  /********************************/

  /* For normal distribution */
  double normal_x;
  double normal_y;
  double normal_rho;
  int normal_is_valid; /* = 0; */
} THGenerator;

#define torch_Generator "torch.Generator"

/* Manipulate THGenerator objects */
TH_API THGenerator * THGenerator_new(void);
TH_API THGenerator * THGenerator_copy(THGenerator *self, THGenerator *from);
TH_API void THGenerator_free(THGenerator *gen);

/* Checks if given generator is valid */
TH_API int THGenerator_isValid(THGenerator *_generator);

/* Initializes the random number generator from /dev/urandom (or on Windows
platforms with the current time (granularity: seconds)) and returns the seed. */
TH_API unsigned long THRandom_seed(THGenerator *_generator);

/* Initializes the random number generator with the given long "the_seed_". */
TH_API void THRandom_manualSeed(THGenerator *_generator, unsigned long the_seed_);

#endif

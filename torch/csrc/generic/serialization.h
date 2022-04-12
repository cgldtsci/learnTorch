#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/serialization.h"
#else

void THPTensor_(writeMetadataRaw)(THTensor *self, int fd);
void THPStorage_(writeFileRaw)(THStorage *self, int fd);
THStorage * THPStorage_(readFileRaw)(int fd);
THTensor * THPTensor_(newWithMetadataFileRaw)(int fd, THStorage *storage);

#endif
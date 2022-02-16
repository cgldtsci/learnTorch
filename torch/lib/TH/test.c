#include <stdio.h>

void defaultTorchErrorHandlerFunction(const char *msg, void *data);

int main() {
    printf("Hello, World!\n");
    defaultTorchErrorHandlerFunction(NULL, NULL);
    return 0;
}

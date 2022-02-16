#include <iostream>

using namespace std;

/* Torch Error Handling */
void defaultTorchErrorHandlerFunction(const char *msg, void *data);

int main(int args, char const *argv[]){
    cout << 'a' << endl;
    defaultTorchErrorHandlerFunction(NULL, NULL);
}
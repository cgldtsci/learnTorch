#include <iostream>
using namespace std;

#include <Python.h>
#include <structmember.h>

#include <stdbool.h>
#include <TH/TH.h>
#include "THP.h"

#include "generic/Storage.cpp"
#include <TH/THGenerateAllTypes.h>

int main(int args, char const *argv[]){
    cout << 'a' << endl;
//    cout << TH_GENERIC_FILE << endl;
    THStorage;
    THPStorage_(newObject)(NULL);
    PyObject *args2 = NULL;
    PyTuple_New(0);
//    THPStorageClass;
}
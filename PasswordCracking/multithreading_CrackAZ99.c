#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <crypt.h>
#include <unistd.h>
#include <semaphore.h>
#include <pthread.h>

/******************************************************************************
  Demonstrates how to crack an encrypted password using a simple
  "brute force" algorithm. Works on passwords that consist only of 2 uppercase
  letters and a 2 digit integer.

  Compile with:
    cc -o CrackAZ99 CrackAZ99.c -lcrypt

  If you want to analyse the output then use the redirection operator to send
  output to a file that you can view using an editor or the less utility:
    ./CrackAZ99 > output.txt

  Dr Kevan Buckley, University of Wolverhampton, 2018 Modified by Dr. Ali Safaa 2019
******************************************************************************/

int count = 0; // A counter used to track the number of combinations explored so far
int threadCount;
// int loopCount = 67600;
int loopCount = 26;

struct threadInfo
{
  int start;
  int end;
};

char startChar, endChar;

char *salt_and_encrypted;

sem_t sem;

/**
 Required by lack of standard function in C.   
*/

void substr(char *dest, char *src, int start, int length)
{
  memcpy(dest, src + start, length);
  *(dest + length) = '\0';
}

/**
 This function can crack the kind of password explained above. All combinations
 that are tried are displayed and when the password is found, #, is put at the 
 start of the line. Note that one of the most time consuming operations that 
 it performs is the output of intermediate results, so performance experiments 
 for this kind of program should not include this. i.e. comment out the printfs.
*/

void *crack(void *args)
{
  int x, y, z;   // Loop counters
  char salt[7];  // String used in hashing the password. Need space for \0 // incase you have modified the salt value, then should modifiy the number accordingly
  char plain[7]; // The combination of letters currently being checked // Please modifiy the number when you enlarge the encrypted password.
  char *enc;     // Pointer to the encrypted password

  char ascii_to_char; // to convert the ASCII int into ASCII char value

  substr(salt, salt_and_encrypted, 0, 6);

  struct threadInfo *tI = (struct threadInfo *)args;
  int startLimit = tI->start;
  int endLimit = tI->end;

  sem_wait(&sem);

  printf("Looping from ASCII start value: %d\n", startLimit);
  printf("Looping to ASCII end value: %d\n", endLimit);

  for (x = startLimit; x <= endLimit; x++)
  {
    ascii_to_char = x;
    for (y = 'A'; y <= 'Z'; y++)
    {
      for (z = 0; z <= 99; z++)
      {
        sprintf(plain, "%c%c%02d", ascii_to_char, y, z);
        enc = (char *)crypt(plain, salt);
        count++;
        if (strcmp(salt_and_encrypted, enc) == 0)
        {
          printf("#%-8d%s %s\n", count, plain, enc);
          // printf("\nEncrypted password found!");
          // printf("\nNow exiting...\n");
          // exit(0);
          // return; //uncomment this line if you want to speed-up the running time, program will find you the cracked password only without exploring all possibilites
        }
        // else
        // {
        //   printf("%-8d%s %s\n", count, plain, enc);
        // }
      }
    }
  }

  sem_post(&sem);

  pthread_exit(0);
}

// preparing the slicelist
void prepareSliceList()
{
  int sliceList[threadCount];
  int remainder = loopCount % threadCount;

  // to store the sliced/divided number of records to be processed by each thread
  for (int i = 0; i < threadCount; i++)
  {
    sliceList[i] = loopCount / threadCount;
  }

  // to update the sliced/divided number of records, that each thread
  // has to process without leaving any record unprocessed/unchecked
  for (int j = 0; j < remainder; j++)
  {
    sliceList[j] = sliceList[j] + 1;
  }

  int startList[threadCount];
  int endList[threadCount];

  /*
  * For threads = 3,
  * first thread will loop through the outer loop processing the characters from A through I i.e., first 9 uppercase characters
  * second thread will loop through the outer loop processing the characters from J through R i.e., other 9 uppercase characters
  * third thread will loop through the outer loop processing the characters from S through Z i.e., the remaining 8 uppercase charcters
  *  */
  for (int k = 0; k < threadCount; k++)
  {
    if (k == 0)
    {
      // startList[k] = 0;
      startList[k] = 65; // ASCII value of 'A'
    }
    else
    {
      startList[k] = endList[k - 1] + 1;
    }

    endList[k] = startList[k] + sliceList[k] - 1;

    printf("\nstartList[%d] = %d\t\tendList[%d] = %d", k, startList[k], k, endList[k]);
  }

  struct threadInfo threadDetails[threadCount];

  for (int l = 0; l < threadCount; l++)
  {
    threadDetails[l].start = startList[l];
    threadDetails[l].end = endList[l];
  }

  pthread_t thread_id[threadCount];

  sem_init(&sem, 0, 1);

  printf("\n\nCreating threads and checking for a matching hash...\n");

  salt_and_encrypted = "$6$AS$1KfiBa8/L2ya34kT9qz5dFQ7prBr2G9i4zCcTsNeyRsouY3FFrgFwYu/KWMzmbo7THRWYA8NH4eCHBtMa3K0g.";

  printf("salt_and_encrypted: %s\n", salt_and_encrypted);

  for (int m = 0; m < threadCount; m++)
  {
    pthread_create(&thread_id[m], NULL, crack, &threadDetails[m]);
  }

  for (int n = 0; n < threadCount; n++)
  {
    pthread_join(thread_id[n], NULL);
  }

  sem_destroy(&sem);
}

// function to display a message explaining what and how arguments should be passed
void inputArguments(char *program_name)
{
  fprintf(stderr, "arguments should be in the order as specified:   %s <number of threads>\n", program_name);
  fprintf(stderr, "where number of threads should be > 0\n");
  exit(0);
}

// getArguments() method created to get the command line arguments
void getArguments(int argc, char *argv[])
{
  if (argc != 2)
  {
    inputArguments(argv[0]);
  }

  threadCount = strtol(argv[1], NULL, 10);

  if (threadCount <= 0)
  {
    inputArguments(argv[0]);
  }
}

int main(int argc, char *argv[])
{
  getArguments(argc, argv);
  prepareSliceList();
  // crack("$6$AS$Sx/vVhm835LD51z2E9.iYxiGvLpkhXHtmqYcDwoYxCi0uBSawtwr42MvlM4UZSZopK6tJWhch1.oQTdX3u.HH."); //Copy and Paste your ecrypted password here using EncryptShA512 program
  printf("%d solutions explored\n", count);

  return 0;
}

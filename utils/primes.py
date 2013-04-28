import math

def primes_up_to(max_num):  
  current_primes = []
  for num in range(2, max_num):
        if prime(num, current_primes):
             current_primes.append(num)            
  return
           
def prime(n, current_primes):
  for i in current_primes:
       if n%i==0:
            return False
    
  return True
  
  
print primes_up_to( 25 )


print primes_up_to( 100 )
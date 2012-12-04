# example usage using badwords.txt (not for the easily offended, but seriously, you're from the internet sooo...)


from dameraulevenshtein import dameraulevenshtein as dl_distance
import string

#open the badwords.txt
file = open("badwords.txt", "r")
swear_list = map( string.strip, file.readlines() ) #strips that annoying \n

def isswear( word, max_distance = 1):
    """
    checks if word is a swear word, or a missing spelling of swear word.
    """
    word = word.lower()
    dl = lambda x: dl_distance(x, word) <= max_distance
    return any( map(dl, swear_list) )
    


    
if __name__=="__main__":
    words_to_test = ["boo", "cameron", "pissy", "ashole", "azzhole", "btiching"]
    
    print "max distance = 1"
    for w in words_to_test:
        print w, isswear(w)    
    print
    print "max distance = 2"
    for w in words_to_test:
        print w, isswear(w,2)
        
        
    
    
    
    
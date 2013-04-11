#Cameron Davidson-Pilon, 2012

#iterative solution

def power_set( list ):
	n = len(list)
	for enumerate in range( 2**n ):
	
		subset = []
		i = enumerate
		
		for j in range(n):
			if i%2:
				subset.append( list[j] )
			i = i >> 1
		
		print subset
		

#resursive solution

def power_set( list, called_empty = False ):
	
	if not called_empty:
		print []
	
	n = len(list)
	print list
	
	if n == 1:
		#not 0, as there are n subsets of length 1, but only 1 
		#subset of length 0
		return
	
	for i in range(n):
		power_set( list[:i] + list[i+1:], True )
		
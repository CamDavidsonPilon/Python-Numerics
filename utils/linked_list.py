
def reverse( list_head, previous = None):
    """
    assume .next is present
    """
    if not list_head.next:
        list_head.next = previous
        return
    else:
        reverse( list_head.next, list_head ) 
        list_head.next = previous if previous else None

        
class Linked( object ):
    
    def __init__(self, next, value ):
        self.next = next
        self.value = value


C = Linked( None, "c")
B = Linked( C, "b")
A = Linked( B, "a")
        
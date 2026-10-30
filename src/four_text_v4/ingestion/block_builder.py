from ..contracts import Block

def blocks(body):
    # A neutral transport span. Quotation/signature classification belongs to the model.
    return (Block('B1','body',0,len(body),body),) if body else ()

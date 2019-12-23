# Poisson Image Editing

## **Python** implement for poisson image editing

### Description

An implementation of poisson image editing. Reference to the paper [Poisson Image Editing](http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)

Now the methods *importing gradients* and *mixing gradients* are implemented.

> From my view, *mixing gradients* works better.

![](/SRC/MixingGradients/target.jpg)
![](/SRC/MixingGradients/source.jpg)
![](/RES/MixingGradients/result.png)

### Usage

You can import *poisson.py* and use directly.
> If you are still confused with the following code, referring to *main.py*

```python
from poisson import Poisson

# source is the picture to insert in Omega
# target is the picture of the background
# mask is the picture to indicate Omega
# all the three are import from cv.imread()
Editor = Poisson(source, target, mask)

# setting is the option of method
# now include ImportingGradients and MixingGradients
result = PoissonEditor.edit(setting="ImportingGradients")
```



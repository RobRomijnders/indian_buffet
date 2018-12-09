# Indian Buffet process LVM
In this project, we set out to explore a Latent Variable Model (LVM) using the Indian Buffet process (IBP). The IBP provides a prior over an infinite number of latent features. Of course, we use only a finite subset of these latent features. This finite set feels just like any other latent variable model. However, the fact that the IBP is a process circumvents that we should define the number of features ourselves.

Another way to get intuition for the IBP follows from comparing to the Dirichlet process mixture model (DPMM). In mixture modeling, we find it cumbersome to define the number of mixtures ahead of time. The DPMM defines a prior over an infinite number of clusters, of which we observe only a finite subset. Just like that, the IBP defines a prior over an infinite amount of latent features, of which we observe only a finite subset.

And *another* way to get intuition for the IBP follows from the generative stories of these processes. Under the Dirichlet process lies the Chinese restaurant process, which we discussed in a [previous project](https://robromijnders.github.io/dpm/). We will compare this Chinese restaurant process to the Indian Buffet process. (I wonder why these stories are based on food and restaurants?!?). 

  * In these Chinese restaurants, there often seems to be an infinite amount of tables. When we enter, we would prefer a popular table, but sometimes we just want to sit at a new table. We end up choosing **one** table, which is analogous to a data point choosing **one** cluster.
  * In these Indian restaurants, there often seems to be an infinite amount of dishes. When we walk the buffet, we try some of the popular dishes, but sometimes we also try some of the new dishes. We end up with **multiple** dishes on our plate, which is analogous to a data point having **multiple** latent variables.\


# How to select from an infinite amount of features?

We can think of the IBP as a distribution over sparse matrices. Those names might scare you, but we have seen it before. In the Dirichlet process mixture model, every data point belonged to one cluster. We can thus think of a sparse matrix, <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/5b51bd2e6f329245d425b8002d7cf942.svg" align=middle width=12.351075000000002pt height=22.381919999999983pt/>, where each row corresponds to a data sample. Each row in this matrix has only one non-zero entry, which corresponds to its cluster. Then its parameters follow from:
<p align="center"><img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/da37fed89884a3ecedd73ae83ec24f1e.svg" align=middle width=61.365809999999996pt height=11.190893999999998pt/></p>
Each row in <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align=middle width=12.282765000000003pt height=22.381919999999983pt/> corresponds to the parameters of one cluster. The matrix <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/5b51bd2e6f329245d425b8002d7cf942.svg" align=middle width=12.351075000000002pt height=22.381919999999983pt/> *selects* one of these rows with its non-zero entry. A matrix now may look like:
![matrix_dpmm](matrix_dpmm.png?raw=true)

Now we extend this equation to the IBP. In the IBP, each row of <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/5b51bd2e6f329245d425b8002d7cf942.svg" align=middle width=12.351075000000002pt height=22.381919999999983pt/> can have multiple non-zero entries. Again, each row in <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg" align=middle width=12.282765000000003pt height=22.381919999999983pt/> corresponds to a latent feature. In that way, each row in <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/5b51bd2e6f329245d425b8002d7cf942.svg" align=middle width=12.351075000000002pt height=22.381919999999983pt/> defines a combination of some latent features that together make up the data sample.
![matrix_ibp_generative](matrix_ibp.png?raw=true)


# Generative story
The generative story of the IBP involves again a metaphor with the Indian restaurant. A generative story encapsulates how we think that the data gets generated. For the IBP follows:

  * A first customer enters the buffet and picks a serving from the first <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/aca8c8df07e723a50f74fb84355dea28.svg" align=middle width=82.77423pt height=24.56552999999997pt/> dishes
  * The n-th customer now enters the buffet and picks a serving according to the popularity of each dish. In other words, if <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/8249cb78ba370605835603be00f4a356.svg" align=middle width=21.618135pt height=14.102549999999994pt/> out of <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/55a049b8f161ae7cfeb0197d75aff967.svg" align=middle width=9.830040000000002pt height=14.102549999999994pt/> former customers tried this dish, then our n-th customer tries this dish with <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/bd945c283ac0d2cb59195652c6b2b92d.svg" align=middle width=107.457075pt height=24.56552999999997pt/>. At the end of the buffet, he tries <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/3abf2d9ee6ca2aedaa75b8533f4bc9e0.svg" align=middle width=84.72816pt height=24.56552999999997pt/> new dishes. 

After we generated data in such manner for all customers, our sparse matrix, <img src="https://github.com/RobRomijnders/indian_buffet/blob/master/svgs/5b51bd2e6f329245d425b8002d7cf942.svg" align=middle width=12.351075000000002pt height=22.381919999999983pt/>, may look like this:
![generative_story](matrix_ibp_generative.png?raw=true)

# Inference
So how would we infer the parameters of a model with infinite amount of latent features? We can iterate over the data, and sample the latent features one by one. We call this Gibbs sampling. There exist many speed ups for this Gibbs sampling. [This documents](http://www.david-andrzejewski.com/publications/llnl-accelerated-gibbs.pdf) accutely describes the details of the Gibbs sampler.

# Data and experiment
A possible problem that got me curious about the IBP concerns dictionary learning. In dictionary learning, we try to decompose the data as a combination of elements from a dictionary. In computer vision, this translates to us searching for elementary patterns by which we can decompose an image. 

As we are dabbling in computer vision now, we use the Lena image. This image is popular in foundational computer vision papers who write about image processing. This is the image:
![lena](lena_image.png)

# Results
After fitting the Indian Buffet process associated with real valued latent factors, we find the following results:

The elementary patterns inferred:

The reconstructed image:

## What do we observe from the results
  
  * First, the reconstructed image looks much like the original image. 
  * Second, the inferred patterns represent elementary patterns like edges and some of the early Gabor filters.

As a personal note: I find it remarkable that a real world image can be reconstructed by this small subset of filters. Moreover, I appreciate the IBP in that I didn't specify the number of patterns. With the IBP, we imagine an infinite set of patterns. During inference, we find the finite set of latent factors for which we have enough evidence.

# Read more


  * [Our project on Dirichlet Process mixture models](https://robromijnders.github.io/dpm/)
  * [Describing the inference in IBP with a collapsed Gibbs filter](http://www.david-andrzejewski.com/publications/llnl-accelerated-gibbs.pdf)
  * [Useful lecture slides by Zoubin Gharamani](https://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture3.pdf)
  * [Lecture as MLSS 2015: Ryan Adams, Dirichlet Processes and friends](https://www.youtube.com/watch?v=xusN7RqKpPI)

Credits

  * Images of the sparse matrices come from [here](https://www.eurandom.tue.nl/events/workshops/2010/YESIV/Prog-Abstr_files/Ghahramani-lecture3.pdf)
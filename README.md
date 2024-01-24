### Problem description

- E-Commerce applications provide an added advantage to customers to buy a product with added suggestions in the form of reviews.

- Obviously, reviews are useful and impactful for customers who are going to buy the products.

- But these enormous amounts of reviews also create problems for customers as they are not able to segregate useful ones.

- Regardless, these immense proportions of reviews make an issue for customers as it becomes very difficult to filter informative reviews.

### Problem description from data science perspective

- To develop a solution for this problem of ranking reviews I have been using a pairwise ranking approach that ranks reviews based on their relevance with the product and rank down irrelevant reviews.

### Solution description

- I developed the solution to this problem in four phases:

  1. Data preprocessing (which includes Language Detection, Gibberish Detection, Profanity Detection)

  2. Feature extraction

  3. Pairwise review ranking

  4. Classification.

- Outcome of the model is a list of reviews for a particular product ranking on the basis of relevance using a pairwise ranking approach.

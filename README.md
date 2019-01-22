# News-Article-LDA
This is a basic (unsupervised) LDA model using Python's gensim. The majority of the work is in preparing the data for the LDA model; news article data can be extremely messy because of the scraping process, as well as because of irrelevant terms that may skew identified topics and keywords. 

# Observed Performance 
Using a dataset of 4776 news articles collected from the top 12 US news publications about Airbnb, was able to distinguish the following topics + keywords (with a default of 10 topics in the model, 6 displayed and 10 keywords per displayed topic): <br></br>
Topic 0: <br></br>
0.021*"trump" + 0.021*"said" + 0.018*"state" + 0.015*"presid" + 0.013*"tax" + 0.011*"bill" + 0.010*"would" + 0.010*"polit" + 0.010*"hous" + 0.010*"support" <br></br>
Topic 1: <br></br>
0.016*"travel" + 0.008*"trip" + 0.008*"hotel" + 0.007*"day" + 0.006*"com" + 0.006*"tour" + 0.005*"restaur" + 0.005*"flight" + 0.005*"one" + 0.005*"like" <br></br>
Topic 2: <br></br>
0.013*"internet" + 0.013*"appl" + 0.012*"facebook" + 0.012*"googl" + 0.011*"media" + 0.011*"app" + 0.011*"ad" + 0.010*"post" + 0.009*"user" + 0.009*"amazon" <br></br>
Topic 3: <br></br>
0.052*"airbnb" + 0.030*"rental" + 0.023*"hotel" + 0.021*"citi" + 0.018*"home" + 0.016*"rent" + 0.016*"said" + 0.015*"host" + 0.012*"new" + 0.012*"term" <br></br>
Topic 4: <br></br>
0.011*"one" + 0.010*"peopl" + 0.009*"say" + 0.008*"time" + 0.007*"go" + 0.007*"year" + 0.007*"like" + 0.006*"would" + 0.006*"get" + 0.006*"think" <br></br>
Topic 5: <br></br>
0.019*"home" + 0.016*"said" + 0.012*"hous" + 0.009*"live" + 0.009*"citi" + 0.008*"build" + 0.008*"year" + 0.007*"park" + 0.007*"rent" + 0.007*"space"

    
# Note about Next Steps:
I am currently looking into semi-supervised LDA models as described in the following resources:<br></br>
http://www.cs.cmu.edu/~bbd/Ramnath-ecml-paper.pdf<br></br>
https://medium.freecodecamp.org/how-we-changed-unsupervised-lda-to-semi-supervised-guidedlda-e36a95f3a164<br></br>
as well as reducing the corpus down to nouns or specific part of speech tags using spaCy, as found in:<br></br>
http://aclweb.org/anthology/U15-1013



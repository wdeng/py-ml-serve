#      Covera NLP        #

### Building a question-answering API

Hello! On behalf of Covera Health, we are excited that you are considering joining our team!

Part of your interview process consists of designing and building out an NLP service.

This document contains detailed instructions on how to get started. Please take some time to read these instructions carefully.

We really hope that you will enjoy working on this exercise with us; it's a small glimpse into what we do here on a day-to-day basis! Have fun! :)

## Getting started

### How the challenge will work:

We will share this repo with you __3 days before__ your submission date. Please PR your code and make sure you commit
any code you intend to execute or would like to show us. Also please email recruiting and let them know when you are done. 

### Build an NLP system
We would like you to design and build an NLP service that is capable of:
* accepting text and a list of questions
* cleaning the text
* processing the text into sentences
* and returning answers to the questions that were asked

We would  like you to allow for the following end points:
* question-answering
* clean-text: accepts text and removes HTML tags and [DEID] tags
* sentencize-text
* clean-and-sentencize

We would like you to expose the individual endpoints so scientists and other developers can use it for their workflows.

We would also like you to show us how you would :
* Test your system? Are there specific tests you would carry out?
* How would you package your system? Please have your docker file ready.
* How would you deploy your system? Please submit your deployment script.
* How would you scale your system? Please submit your Kubernetes manifest.

Lastly we recommend using open source packages and models such torch and huggingface transformers as this is what 
matches our stack. Using a sandbox tool such as [minikube](https://minikube.sigs.k8s.io/docs/start/) to test out your 
deployment may also be very helpful.


##### The input payload to your `question-answering` should look like the following examples:

Example: 1
```python
{
    'text': """<HTML><HEAD>Steam engines are external combustion engines, <INSIDE> where the working fluid is separate from the combustion 
products. Non-combustion heat sources such as solar power, nuclear power or geothermal energy may be used. The ideal 
thermodynamic cycle used to analyze this process is called the Rankine cycle. In the cycle, water is heated and 
transforms into steam within a boiler operating at a high pressure. When expanded through pistons or turbines, mechanical 
work is done. The reduced-pressure steam is then condensed and pumped back into the boiler.</BODY></HTML>""",
    'question': ["Along with geothermal and nuclear, what is a notable non-combustion heat source?",
 "What ideal thermodynamic cycle analyzes the process by which steam engines work?"]
}
```


Example 2:
```python
    'text': """  No spinal cord demyelination. Spinal MRI within normal limits.  Right cerebellar hemangioblastoma again seen Approved and Electronically 
Signed by: [AWS-DEID]  [AWS-DEID]  [DEID] [DEID] [DEID] [DEID] [DEID]""",
    'question': ["What kind of scan is this?",
 "Where is the hemangioblastoma?"]
```

##### The output from the `question-answering` service should look like:
```python
[{
    "answer": "This is an MRI scan.",
    "score": 0.05684709993458714,
    "metadata": {
        "compute_time": 0.8131041526794434,
        "model_version": "roberta",
        "status_code": 200}
},
{
    "answer": "The hemangioblastoma is in the right cerebellum.",
    "score": 0.05684709993458714,
    "metadata": {
        "compute_time": 0.8131041526794434,
        "model_version": "roberta",
        "status_code": 200}
}]
```


##### We are looking to see:
* Code quality
* What tools/framework do you use?
* How you test your system?
* What design decisions you make?
* If your system breaks how would you troubleshoot?
* How would you scale in terms of volume?
* If you had three weeks instead of three days what would you do differently or in addition to what you did?

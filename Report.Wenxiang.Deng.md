#      Excersize Report        #

### Building a question-answering API

Hello! On behalf of Covera Health, we are excited that you are considering joining our team!

Part of your interview process consists of designing and building out an NLP service.

This document contains detailed instructions on how to get started. Please take some time to read these instructions carefully.

We really hope that you will enjoy working on this exercise with us; it's a small glimpse into what we do here on a day-to-day basis! Have fun! :)

## Build an NLP system
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

## Deployment

### Dockerize

The Dockerfile consists of two parts,
1. 
2. 
3. Alternatively, Because the NLP models usually changes more frequently than inference code. To achieve this,  For the scope of the excersize, this is not implemented.


p90 and p99 latency

### setup k8s

Here I use EKS 1.25 for the test deployment.
```
eksctl create cluster --name will-test --region us-west-2 --with-oidc \
--ssh-access --ssh-public-key eks-ml-key --managed

aws eks update-kubeconfig --region us-west-2 --name will-test
```

The node groups can be set in `EKS > Clusters > will-test > Compute`. Here I used 3 t3.medium instance.

For deployment
```
k create -f ./k8s/application.yaml
```

### QA
* What tools/framework do you use?
* How you test your system?
* What design decisions you make?
* If your system breaks how would you troubleshoot?
* How would you scale in terms of volume?
  1. Loadbalancing and Ingress
  1. Horinzontal scaling: AutoScaler
  2. possible caching: if some of the requests and payloads are 
  4. event driven: 
* If you had three weeks instead of three days what would you do differently or in addition to what you did?
  1. interact with stakeholders for testing
  1. put all the resources behind
  2. implement CD pipeline with ArgoCD or Flux in k8s cluster, and add QA process after initial production deployment
  2. because such APIs are usually behind webservices grpc 
  3. monitoring system with Prometheus and logging systems with elastic stack or 
  4. further optimize the ONNX pipeline for faster inference

1. Unit testing, I did unit test locally since it's much faster, but it can be a part of CI pipeline; 
2. run the local server and use multithreading to test the whole server; 3. build docker image and test on dev k8s server.


Complete CI/CD pipeline 

Logging system is currently not production ready. Instead of persistenting the Pod level logs (currently in `app.log` file), I would use Datadog managed service, or considering setting up open source framework with Loki + Grafana. Elasticsearch is another option, but since it take a lot more effort to manage, I wouldn't consider it at the moment.

Beyond 3 weeks, and for a more production ready setup, there are other things

Move CI to managed hosts in AWS, and move docker images to ECR
setup model; In the current excersize, the CI building takes more than 5 minutes for the Github Registry image, since there is no caching for each run. Build a better base image.

If in practice, there are online features (prediction features effective for certain period of time), I would consider integrating a feature store with databases or open source framework like Feast.
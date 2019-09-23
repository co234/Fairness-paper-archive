# Fairness

**Why do we care?**

> Nowadays, many things have automated by ML systems. For instance, companies use ML system to help them select job applicants; courts in US use ML algorithm for recidivism prediction; Amazon, Netflix use recommender system...
>
> We care about fairness because it is highly related to our own benefits.





### Cause of unfairness

- Sample size disparity

  According to the law of large numbers, with more training examples, the empirical risk is closer to the expected risk, which means increasing training samples, it will help to learn the best hypothesis and therefore get smaller generalisation error. However, sample size in the minority group and majority group is highly imbalanced, which lead to higher error rate in minority group.

- Bias in data

  - Collection:

    Demographic, geographic, behavioral, temporal biases

  - Measurement:

    What do we choose to measure? How do we measure?

  - **Pre-existing biases**:

    Gender roles in text and images, racial stereotypes in language...

    > ...Systems trained to make decisions based on historical data will naturally inherit the past biases....



### Fairness criteria

**Typical Setup:**

<span style="color:red">**X**</span>: features of an individual

<span style="color:red">**A**</span>: Sensitive attribute (ex. gender, race)

<span style="color:red">**C=C(X,A)**</span>: classifier mapping X and A to some prediction

<span style="color:red">**Y**</span>: actual outcome



However.....

<span style="color:red">**X**</span>: incorporates all sorts of measurement biases

<span style="color:red">**A**</span>: Often not even known, ill-defined, misreported, inferred

<span style="color:red">**C=C(X,A)**</span>: Often not well defined (ex. large production ML system)

<span style="color:red">**Y**</span>: often poor proxy of actual variable of interest



**Some definitions:**

- Demographic parity:

  > Assume $$C \in \{0,1\}, A\in\{0,1\}$$, Classifier $$C$$ satisfies demographic parity if:
  >
  > ​										$$P(C=1\mid A=1) = P(C=1\mid A=0)$$

- Accuracy parity:

  > Assume $$A\in \{0,1\}$$, Classifier $$C$$ satisfies accuracy parity if:
  >
  > ​										$$P(C=Y\mid A=1)=P(C=Y\mid A=0)$$

- Precision parity

  > Assume $$C \in \{0,1\}, A\in\{0,1\}$$, $$Y \in \{0,1\}$$, Classifier $$C$$ satisfies demographic parity if: 										
  $$P(Y=1\mid C=1, A=1) = P(Y=1\mid C=1,A=0)$$

- True positive parity

  > Assume $$C \in \{0,1\}, A\in\{0,1\}$$, $$Y \in \{0,1\}$$, Classifier $$C$$ satisfies demographic parity if: 										
  $$P(C=1\mid Y=1, A=1) = P(C=1\mid Y=1,A=0)$$

- **Group fairness**:

  > Group fairness also known as statistical parity, ensures that the overall proportion of members in a protected group receiving positive (negative) classification are identical to the proportion as whole.

- **Individual fairness**:

  > ..Any two individuals who are similar with respect to a particular task should be classified similarly.



**Trade-offs**

A classifier C cannot simultaneously achieve precision parity, true positive parity and false positive parity unless:

							- $$C=Y$$ (the classifier is perfect) or
							- $$P(Y=1 \mid A=0) = P(Y=1 \mid A=1)$$ (base rates are equal)



Beyond observational measures is causality.

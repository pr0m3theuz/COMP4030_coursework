**COMP4030 / DATA SCIENCE WITH MACHINE LEARNING COURSEWORK 2025-26 COURSEWORK BRIEF** 

| Assessment Name  | Coursework – Data Science Study  | Weight  | 75% |
| :---- | :---- | :---: | :---- |
| **Description and   Deliverable(s)** | This assignment requires you to work individually. You will need to make use of what you have learned in  order to produce a series of deliverables regarding your  machine learning project:  • A **poster** or a **slide deck**.  • A short paper (up to 4 pages) using the **IEEE  template for formatting.**  • A single runnable and commented **Jupyter  notebook**. |  |  |
| **Release Date**  | Tuesday 3rd March 2026 |  |  |
| **Submission Date**  | Tuesday 5th May 2026 by 3:00 PM |  |  |
| **Late Policy**  | As per University of Nottingham’s default late policy,  work submitted after the deadline will be subject to a  penalty of 5 marks for each late working day out of the  total 100 marks.  Late submission deadline is Monday 11th May 2026 3:00  PM, due to needing to schedule presentations. Submissions after this date will only be accepted through  **the extenuating circumstances process**1. |  |  |
| **Feedback Mechanism  and Date**  | We will aim to provide feedback as early as possible in  Moodle (typically within 15 working days, outside of exam  period). |  |  |

**1\. INTRODUCTION** 

This document describes the main assessment of COMP4030, which is an **individual coursework** weighted at 75% of the module grade. You will produce a data science/machine  learning study in which you will select a dataset, perform a series of experiments in order to  answer one or more carefully crafted research questions, and then report on your work in a  format of your choice (either a presentation, or a poster). 

**2\. INSTRUCTIONS** 

For this coursework assignment you will need be required to conduct a data science project from  start to finish. You will produce the following deliverables: 

1 See https://student-enquiries.nottingham.ac.uk for more details  
(1) You can choose to produce either a **research poster** (size A0, horizontal format) or a  **slide-based presentation** (e.g., PowerPoint) as a medium to present the work in the  poster paper (guidance will be given on the design of research posters). The submission  will need to be in a PDF format. 

(2) **A short paper** (up to 4 pages) explaining your work, submitted as a PDF file. (3) A well-commented **Jupyter notebook** containing your experiments, as an ipynb file. 

You will then have to present your poster to a member of the module delivery team (module  convenors and TAs) in a 10 \+ 5-10 minutes format (10 minutes of presentation, 5-10 minutes of  questions and answers), between the 03/05/2026 and the 15/05/2026. There are more details  about each of the deliverable in the following subsections. 

**2.1. The poster or presentation** 

You can choose to either produce a poster or a slide-based presentation. 

Research poster specification 

Your poster should be A0 size, horizontally orientated. It should contain the main elements of  your work – introduction to the problem, objectives, methodology, results, and discussion.  

Presentation slides specification 

Your slides should be in a PDF format. They should contain the main elements of your work – introduction to the problem, objectives, methodology, results, and discussion.  

Design 

While you are not marked on the design of your poster or slides, you are expected to make  sensible use of design elements such as plots, diagrams, and other visual aids. You will need to  present your poster during the presentation slot so you should make sure that it supports your  study. You should also make sure everything is readable at a reasonable distance, on a large  enough screen such as the ones in the pods or in meeting rooms.  

**2.2. The short paper** 

Your paper should be up to 4 pages (**including** tables, diagrams but excluding references) and submitted as a PDF. There is no minimum page limit, although we would be surprised if you  managed to fit everything in fewer than 2 pages. The diagrams table and diagrams should add  value to the writing and not just be used for decoration. Your paper should be organised into the  following sections (some variation is allowed, but we recommend sticking to what works): 

**1\. Title** 

**2\. Abstract** 

**3\. Introduction** to the dataset and research question(s) and exploratory analysis **4\. Related works** 

**5\. Methodology** – including an evidence-based justification for your approaches **6\. Results** – data analysis, pre-processing and prediction 

**7\. Discussion** – discussing results both within the study and with the literature **8\. Conclusion** 

**9\. References** 

**2.3. Jupyter Notebook** 

Your code should be integrated as a **single** Jupyter Notebook. We should be able to run this from  beginning to end to generate your results in addition to the paper, so extra care should be given  to reproducibility (reproducibility guidelines will be mentioned below and elaborated upon in the   
module). For example, make sure that all the packages you are using are declared early in your  script. Testing will be done using whatever version of Python is available in A32 and/or the Virtual  Desktop. Submission of another format than Jupyter notebook will result in a mark of 0 for this  component. We should be able to run your project on a reasonable machine, so avoid overly  complex models as much as possible. 

The ultimate aim of this coursework is to give you first-hand experience on working with a  relatively real data set, your code needs to reflect your learning for the entire process, from the  first stages of data science: data preparation and pre-processing, exploratory data analysis, to  the later stages of knowledge extraction, learning, prediction, and evaluation. 

**3\. MARKING CRITERIA** 

You will be marked based on all components (Paper, Code, Presentation) taken as a whole, so  skipping any one of them will result in a failing grade.

|  | Criterion  | Assessment Indicators  |
| :---- | :---: | ----- |
| **A. Paper**  | **Structure &   Compliance** | • Adherence to the 4-page limit and template.  • Presence of clear citations and bibliography.  • Professional formatting of figures and tables. |
|  | **Technical   Content** | • Explicit statement of the research problem.  • Logical justification for selection of approaches and  models.  • Objective and critical (not just descriptive) analysis of  results.  • Insightful discussion and contextualisation of the work within the literature and the end application. |
| **B. Code** | **Reproducibility** | • Sequential execution (*Run All*) works without error. • Inclusion of necessary data files or correct paths. • Presence of Markdown cells to explain the logic. |
|  | **Implementation  quality** | • Correct application of validation methods.  • Absence of data leakage in the pipeline.  • Use of appropriate evaluation metrics (e.g., F1, ROC). |
| **C. Presentation** | **Visual aids  (Poster/Slides)** | • High visibility of text and graphics.  • Appropriate use of charts.  • Professional design aesthetics. |
|  | **Oral delivery** | • Strict adherence to the time constraint.  • Coherent narrative structure (Intro, Method, Result,  Conclusion).  • Use of precise technical vocabulary. |
|  | **Questions &  answers** | • Accuracy in technical answers.  • Defence of design and technical choices against  alternatives.  • Discussion of limitations where appropriate. |

**4\. COURSEWORK POLICIES** 

**4.1. Reproducibility** 

It should be straightforward to rerun your experiments straight from your Python notebook. In  order to do so, please use the reproducibility checklist (adapted and simplified from the AAAI  conference instructions) to check your work: 

• Any code required for pre-processing data is included or accessible programmatically. • All source code required for conducting and analysing the experiments is included. 

• All source code implementing new methods have comments detailing the  implementation, with references to the paper where each step comes from. 

• If an algorithm depends on randomness, then the method used for setting seeds is  described in a way sufficient to allow replication of results. 

• This paper describes evaluation metrics used and explains and justifies the motivation  for choosing these metrics. 

• This paper states the number of runs used to compute each reported result. 

• Analysis of experiments goes beyond single-dimensional summaries of performance  (e.g., average; median) to include measures of variation, confidence, or other  distributional information. 

• The significance of any improvement or decrease in performance is judged using  appropriate statistical tests (e.g., Wilcoxon signed-rank test). 

• This paper lists all final (hyper-)parameters used in the experiments. 

• This paper states the number and range of values tried per (hyper-) parameter during  development of the paper, along with the criterion used for selecting the final parameter  setting. 

**4.2. Plagiarism and generative AI** 

Generative AI, mostly in the form of large language models, have become embedded in several  applications and increasingly hard to avoid. For that reason, we try to make our generative AI  policy as clear as possible: 

| What is allowed:  | What is not allowed |
| :---- | :---- |
| Where it’s unavoidable (e.g., embedded in  search engine).  As part of assisted coding (code completion,  bug finding).  To help you understand complex topics. | To write the report in your place (even if you  rephrase it later on)  To write the code for you (even if you reshape  it afterwards). |

Our plagiarism policy is simple: **do not plagiarise**. Plagiarism cases (whether based on AI, other  work, or your own, uncited prior work) will be sent to the academic misconduct committee. To  avoid accidental plagiarism: 

• Cite the work you’re getting your inspiration from using a consistent referencing system  (we recommend IEEE for the sake of simplicity, but we are not attached to it) • Reference the code you’re using. Preferably in the comment where you’re using it, but  potentially in a text cell that explains it.
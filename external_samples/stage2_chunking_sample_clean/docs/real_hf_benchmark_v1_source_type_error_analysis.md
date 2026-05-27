# Real HF benchmark v1 source_type error analysis

## Scope

This report analyzes `source_type` errors on `real_hf_benchmark_v1` using `pseudo_gold_v2` and the existing v2 prediction outputs. It does not change classifiers, taxonomy, thresholds, chunks, or labels.

## Key finding

Full-label accuracy is low mainly because `source_type` is a separate unresolved task. Taxonomy v2 improved semantic `domain/field/subfield`, but it did not improve source_type because the semantic classifiers do not really predict source_type.

## Expected source_type distribution

- `educational`: 39
- `reference`: 35
- `math`: 15
- `web_general`: 9
- `news`: 7
- `forum_qa`: 5
- `commercial_product`: 4
- `boilerplate_or_noise`: 2
- `code`: 2
- `wiki_reference`: 1
- `legal_government`: 1

## Predicted source_type distributions

### rule

- Accuracy: 0.1833
- Per dataset: {'FineMath': 0.25, 'FineWeb': 0.05, 'FineWeb-Edu': 0.25}
- Predicted distribution:
  - `unknown`: 77
  - `educational`: 37
  - `math`: 4
  - `commercial_product`: 2
- Top mismatch groups:
  - `reference` -> `unknown`: 25
  - `educational` -> `unknown`: 19
  - `math` -> `unknown`: 10
  - `reference` -> `educational`: 9
  - `web_general` -> `unknown`: 6
  - `news` -> `unknown`: 4
  - `commercial_product` -> `unknown`: 4
  - `forum_qa` -> `unknown`: 4
  - `web_general` -> `educational`: 3
  - `news` -> `educational`: 3

### lexical

- Accuracy: 0.0750
- Per dataset: {'FineMath': 0.1, 'FineWeb': 0.075, 'FineWeb-Edu': 0.05}
- Predicted distribution:
  - `unknown`: 106
  - `educational`: 10
  - `boilerplate_or_noise`: 2
  - `forum_qa`: 1
  - `news`: 1
- Top mismatch groups:
  - `reference` -> `unknown`: 35
  - `educational` -> `unknown`: 28
  - `math` -> `unknown`: 15
  - `web_general` -> `unknown`: 9
  - `news` -> `unknown`: 6
  - `forum_qa` -> `unknown`: 5
  - `commercial_product` -> `boilerplate_or_noise`: 2
  - `boilerplate_or_noise` -> `unknown`: 2
  - `commercial_product` -> `unknown`: 2
  - `code` -> `unknown`: 2

### minilm

- Accuracy: 0.0000
- Per dataset: {'FineMath': 0.0, 'FineWeb': 0.0, 'FineWeb-Edu': 0.0}
- Predicted distribution:
  - `unknown`: 120
- Top mismatch groups:
  - `educational` -> `unknown`: 39
  - `reference` -> `unknown`: 35
  - `math` -> `unknown`: 15
  - `web_general` -> `unknown`: 9
  - `news` -> `unknown`: 7
  - `forum_qa` -> `unknown`: 5
  - `commercial_product` -> `unknown`: 4
  - `boilerplate_or_noise` -> `unknown`: 2
  - `code` -> `unknown`: 2
  - `wiki_reference` -> `unknown`: 1

### hybrid

- Accuracy: 0.1833
- Per dataset: {'FineMath': 0.25, 'FineWeb': 0.05, 'FineWeb-Edu': 0.25}
- Predicted distribution:
  - `unknown`: 77
  - `educational`: 37
  - `math`: 4
  - `commercial_product`: 2
- Top mismatch groups:
  - `reference` -> `unknown`: 25
  - `educational` -> `unknown`: 19
  - `math` -> `unknown`: 10
  - `reference` -> `educational`: 9
  - `web_general` -> `unknown`: 6
  - `news` -> `unknown`: 4
  - `commercial_product` -> `unknown`: 4
  - `forum_qa` -> `unknown`: 4
  - `web_general` -> `educational`: 3
  - `news` -> `educational`: 3

## Confusion matrix: hybrid_v2 source_type

| expected \ predicted | `commercial_product` | `educational` | `math` | `unknown` |
| --- | ---: | ---: | ---: | ---: |
| `boilerplate_or_noise` | 0 | 0 | 0 | 2 |
| `code` | 0 | 0 | 0 | 2 |
| `commercial_product` | 0 | 0 | 0 | 4 |
| `educational` | 1 | 19 | 0 | 19 |
| `forum_qa` | 0 | 0 | 1 | 4 |
| `legal_government` | 0 | 0 | 0 | 1 |
| `math` | 0 | 2 | 3 | 10 |
| `news` | 0 | 3 | 0 | 4 |
| `reference` | 1 | 9 | 0 | 25 |
| `web_general` | 0 | 3 | 0 | 6 |
| `wiki_reference` | 0 | 1 | 0 | 0 |

## Failure pattern examples

### reference predicted unknown (25 records)

#### FineWeb-Edu_000014_000 (FineWeb-Edu)

- Text preview: Human nature and the nature of war Twenty years ago, Canadian military historian and journalist Gwynne Dyer fascinated the public in 45 countries with his award-winning television series on the nature of war. He followed up that success with an equally remarkable book that grew out of the research that went into the television series. Now Dr. Dyer has published, in his words, "a completely rewritten and updated new e...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: book review / geopolitical reference; missing_taxonomy: history/politics/international_relations
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb-Edu_000018_000 (FineWeb-Edu)

- Text preview: The OED “is widely regarded as the accepted authority on the English language. It is an unsurpassed guide to the meaning, history, and pronunciation of 600,000 words— past and present—from across the English-speaking world. As a historical dictionary, the OED is very different from those of current English, in which the focus is on present-day meanings. You’ll still find these in the OED, but you’ll also find the his...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: library database description for OED; language/reference resource
- rule: `unknown/None/None/None`
- lexical: `unknown/multilingual/mixed_language/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000009_043 (FineWeb)

- Text preview: allows communication between the master computer and the data tap. In a preferred embodiment, both transmitted and received communications are differentially driven by utilizing RS-422 or RS-485 interfaces. The use of these interfaces allows effective communications over relatively large distances. That is, an RS-422 or RS-485 interface allows effective communication up to a length of approximately 5,000 feet, wherea...
- Expected: `reference/technology/patents/hardware_networking`
- Review note: technical patent-style system description; missing_taxonomy: patents/hardware_networking; taxonomy_v2_update: software/programming/documentation -> technology/patents/hardware_networking
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/hardware_networking`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000009_051 (FineWeb)

- Text preview: the rental of the first video cassette. These steps are shown by displays 6 and 7 below. ##STR6## Next, the Total key 530 is pressed. As shown below, in Display 8, pressing this key causes the total amount, including tax, to be calculated and displayed. Further, individual subtotals of the rental charge; charges for any sales included in the transaction; and tax amounts also are displayed. ##STR7## Assume, at this po...
- Expected: `reference/technology/patents/POS_systems`
- Review note: technical patent-style POS transaction flow; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/POS_systems`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000009_027 (FineWeb)

- Text preview: At step S90, the "trapper" subroutine is executed. The trapper subroutine of step S90 is shown in detail in FIG. 4B. Referring now to FIG. 4B, the trapper subroutine starts at 125 and proceeds to step S130, in which a flag is detected, if it is present, indicating that video cassette rental data was being selected or "trapped" during the previous interrupt cycle. If the determination in step S130 is affirmative, proc...
- Expected: `reference/technology/patents/POS_systems`
- Review note: patent algorithm/subroutine description; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/hardware_networking`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

### reference predicted educational (9 records)

#### FineWeb-Edu_000022_002 (FineWeb-Edu)

- Text preview: Picasso, Joan Miró, and Salvador Dalí talked on the radio in his place to publicize the show. Picasso's friend, the surrealist poet Paul Éluard, traveled down to Barcelona from Paris for the exhibit and delivered a lecture at the opening. Éluard, standing in for Picasso, received the hearty support of students who chanted, "Picasso, the Marxist," confirming that they saw Picasso as a radical. (In fact, although Picas...
- Expected: `reference/humanities/art_history/article`
- Review note: historical art/politics narrative; missing_taxonomy: art_history/political_history; taxonomy_v2_update: reference/encyclopedic_article/general -> humanities/art_history/article
- rule: `educational/education/general_education/article`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineWeb_000015_000 (FineWeb)

- Text preview: |- candidate number||14295| |- NTR Number||NTR3827| |- ISRCTN||ISRCTN wordt niet meer aangevraagd.| |- Date ISRCTN created| |- date ISRCTN requested| |- Date Registered NTR||30-jan-2013| |- Secondary IDs||NL41498.018.12 / 2012_211; CCMO / METC AMC| |- Public Title||Usability of Subcutaneous Continuous Glucose Monitoring in Critically Ill Patients.| |- Scientific Title||Usability of Subcutaneous Continuous Glucose Mon...
- Expected: `reference/science/biology/article`
- Review note: clinical trial registry table; missing_taxonomy: medicine/clinical_trials
- rule: `educational/education/general_education/article`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineWeb_000009_040 (FineWeb)

- Text preview: Upon pressing the function key F10 the transaction currently displayed on the screen is aborted. In accordance with the present invention, upon pressing the "hot keys" on the keyboard, that is, the "Alt" and the left "shift" keys, the main menu of the teacher program is displayed in the shaded area at the bottom of the display as shown in Screen 3 illustrated in FIG. 4G. The main menu allows the installer to select p...
- Expected: `reference/technology/patents/POS_systems`
- Review note: technical software UI/patent procedure; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `educational/education/general_education/article`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineWeb_000009_023 (FineWeb)

- Text preview: Accordingly, each data tap is connected between the server and one of the terminals, through cable 326 or 330. When a printer is connected to a server or terminal, additional parallel connections are used. For example, printers 308 and 310 are respectively connected to the parallel ports of the data taps by cables 322 and 330 and from there to the server 302 and terminal 306 by way of the cables 321 and 329. A progra...
- Expected: `reference/technology/patents/hardware_networking`
- Review note: technical patent/network terminals text; missing_taxonomy: patents/hardware_networking; taxonomy_v2_update: software/programming/documentation -> technology/patents/hardware_networking
- rule: `educational/education/general_education/article`
- lexical: `unknown/technology/patents/hardware_networking`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineWeb-Edu_000014_002 (FineWeb-Edu)

- Text preview: like the UN and the International Court at the expense of national sovereignty. The problem is to convince the peoples of the world and the politicians in charge that this is what we must learn to do. The end of the Cold War has given us a temporary reprieve. But if we do not use this opportunity to strengthen the UN, we may not be able to avoid the doom that surely awaits us. War by Gwynne Dyer, Toronto, 2004, 484 p...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: book review/political argument on war and UN; missing_taxonomy: international_relations
- rule: `educational/education/general_education/article`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

### math predicted unknown or educational (12 records)

#### FineMath_000002_006 (FineMath)

- Text preview: Let y represent theta Prove: 1 + 1/tan^2y = 1/sin^2y My Answer: LS: = 1 + 1/tan^2y = (sin^2y + cos^2y) + 1 /(sin^2y/cos^2y) = (sin^2y + cos^2y) + 1 x (cos^2y/sin^2y) = (sin^2y + cos^2y) + (sin^2y + cos^2y) (cos^2y/sin^2y) = (sin^2y … 8. ### TRIG!
- Expected: `math/stem/mathematics/algebra`
- Review note: trigonometric identity proof; closest taxonomy algebra
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineMath_000002_003 (FineMath)

- Text preview: expres the following as sums and differences of sines or cosines cos8t * sin2t sin(a+b) = sin(a)cos(b) + cos(a)sin(b) replacing by by -b and using that cos(-b)= cos(b) sin(-b)= -sin(b) gives: sin(a-b) = sin(a)cos(b) - cos(a)sin(b) … 3. ### math
- Expected: `math/stem/mathematics/algebra`
- Review note: trigonometry identities; closest taxonomy algebra
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineMath_000022_011 (FineMath)

- Text preview: What can be used to convert inches in 5 3? To quickly and accurately convert inches in 5 3, you can either use a calculator or an online tool such as our Inches To Feet Calculator. This calculator allows you to enter multiple measurements at once so you don’t have to do each one individually. Additionally, it provides detailed instructions on how to calculate feet and fractions of an inch into inches correctly. Is th...
- Expected: `math/stem/mathematics/arithmetic_measurement`
- Review note: unit conversion Q&A; missing_taxonomy: arithmetic/measurement; taxonomy_v2_update: stem/mathematics/algebra -> stem/mathematics/arithmetic_measurement
- rule: `educational/education/general_education/article`
- lexical: `unknown/stem/mathematics/arithmetic_measurement`
- minilm: `unknown/stem/mathematics/arithmetic_measurement`
- hybrid: `educational/stem/mathematics/arithmetic_measurement`

#### FineMath_000022_008 (FineMath)

- Text preview: Inches in 5 3 can also be used as a part of various formulas. For instance, the BMI (Body Mass Index) formula uses height expressed in inches to calculate a person’s body weight. Additionally, many growth charts use inches to track the growth of babies and children over time. Knowing how many inches in 5 3 is an important part of using these formulas correctly. Conclusion: How Many Inches In 5 3
- Expected: `math/stem/mathematics/arithmetic_measurement`
- Review note: measurement conversion and formulas; missing_taxonomy: arithmetic/measurement; taxonomy_v2_update: stem/mathematics/algebra -> stem/mathematics/arithmetic_measurement
- rule: `unknown/None/None/None`
- lexical: `unknown/stem/mathematics/arithmetic_measurement`
- minilm: `unknown/stem/mathematics/arithmetic_measurement`
- hybrid: `unknown/stem/mathematics/arithmetic_measurement`

#### FineMath_000016_004 (FineMath)

- Text preview: The first column calculation: 1.00=sqrt(1) 0.80=0.8/1.00 0.20=0.2/1.00 0.50=0.5/1.00 The second column calculation(numbers rounded two the second decimal): 0.60=sqrt(1-0.80*0.80) 0.57=(0.5-0.2*0.8)/0.6 0.17 = (0.5-0.5*0.8)/0.6 The third column calculation(numbers rounded two the second decimal): 0.80=sqrt(1-(0.20*0.20+0.57*0.57)) 0.38=(0.5-(0.50*0.20+0.17*0.57))/0.80 The last column calculation(numbers rounded two th...
- Expected: `math/stem/mathematics/algebra`
- Review note: matrix/numeric decomposition calculation; closest taxonomy algebra
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

### web_general predicted unknown/educational (9 records)

#### FineWeb_000022_001 (FineWeb)

- Text preview: chills watching that again. Childress was drafted #19 overall in the 1995 Draft by Detroit but only hung around in the league long enough for a cup of coffee. A very small cup. Since leaving the NBA, he has played internationally in Turkey, Australia and Italy. He was a tremendous talent and it honestly bugs me that we weren’t able to see him enjoy a long NBA career. Generously listed at 6’2″, what played against him...
- Expected: `web_general/media/news/article`
- Review note: sports blog/article; missing_taxonomy: sports
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb-Edu_000010_011 (FineWeb-Edu)

- Text preview: tale begins at Camp Nazareth (that’s the name of the overnight camp at the end of our lake). Its run by the local Catholic Diocese which has had little success in recent years attracting enough kids. More often than not, this wonderful facility – it can hold up to 300 kids at any one time - is terribly under used. Fortunately, it seems that they’ve now discovered ways to attract alternate users like family reunions, ...
- Expected: `web_general/education/general_education/article`
- Review note: reflective anecdote about rowing/team learning; missing_taxonomy: sports/coaching
- rule: `educational/education/general_education/article`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineWeb-Edu_000010_008 (FineWeb-Edu)

- Text preview: but clearly remembered moment in time. 9/11 does all of those things and more. My wife and I were in NYC: preparing to get on the George Washington Bridge to go into Manhattan when the first plane hit; coming to a complete stop on the road and in our lives; watching in fear and confusion as the second plane hit; staring in horror as first one and then the other building fell; hearing about the other plane crashes in ...
- Expected: `web_general/media/news/article`
- Review note: personal essay/reflection on 9/11; missing_taxonomy: personal_essay/history
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb-Edu_000010_016 (FineWeb-Edu)

- Text preview: fashioned sayings? Here in New York last week the mayor and the meteorologists got it wrong – but not by much. The winds blew and the rains fell and, though there was less flooding and damage than predicted here, they made damn sure we were prepared by scaring the daylights out of us with their dire warnings. Now some people are complaining because they scared us; but those same people complained when they didn’t sca...
- Expected: `web_general/science/environmental_science/article`
- Review note: weather/storm preparedness personal commentary; missing_taxonomy: weather/personal_blog
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000023_000 (FineWeb)

- Text preview: That day was a watershed in my transition from a talented amateur to a serious artist. I had always subscribed to the mantra "good enough is not good enough" in my profession, but until that day I had not done so as an artist. From then on I resolved that I would make things as good as I could in my art, even if that took time and effort. Which is a long way to get to how I spent yesterday afternoon. Several weeks ag...
- Expected: `web_general/education/general_education/article`
- Review note: personal art/quilting reflection; missing_taxonomy: arts/crafts
- rule: `unknown/None/None/None`
- lexical: `unknown/humanities/art_history/article`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

### patent/POS-like chunks (22 records)

#### FineWeb_000009_043 (FineWeb)

- Text preview: allows communication between the master computer and the data tap. In a preferred embodiment, both transmitted and received communications are differentially driven by utilizing RS-422 or RS-485 interfaces. The use of these interfaces allows effective communications over relatively large distances. That is, an RS-422 or RS-485 interface allows effective communication up to a length of approximately 5,000 feet, wherea...
- Expected: `reference/technology/patents/hardware_networking`
- Review note: technical patent-style system description; missing_taxonomy: patents/hardware_networking; taxonomy_v2_update: software/programming/documentation -> technology/patents/hardware_networking
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/hardware_networking`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000009_051 (FineWeb)

- Text preview: the rental of the first video cassette. These steps are shown by displays 6 and 7 below. ##STR6## Next, the Total key 530 is pressed. As shown below, in Display 8, pressing this key causes the total amount, including tax, to be calculated and displayed. Further, individual subtotals of the rental charge; charges for any sales included in the transaction; and tax amounts also are displayed. ##STR7## Assume, at this po...
- Expected: `reference/technology/patents/POS_systems`
- Review note: technical patent-style POS transaction flow; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/POS_systems`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000009_027 (FineWeb)

- Text preview: At step S90, the "trapper" subroutine is executed. The trapper subroutine of step S90 is shown in detail in FIG. 4B. Referring now to FIG. 4B, the trapper subroutine starts at 125 and proceeds to step S130, in which a flag is detected, if it is present, indicating that video cassette rental data was being selected or "trapped" during the previous interrupt cycle. If the determination in step S130 is affirmative, proc...
- Expected: `reference/technology/patents/POS_systems`
- Review note: patent algorithm/subroutine description; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/hardware_networking`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb_000009_013 (FineWeb)

- Text preview: the data regarding revenue-sharing cassettes from all of the other data it receives. Then, it computes the shares of the revenues due to the cassette owners or distributors and to the retailers. Reports and funds are sent to the respective participants in the revenue-sharing program according to a formula previously agreed upon. If desired, reports can be transmitted through a modem 30 and telephone lines to computer...
- Expected: `reference/technology/patents/POS_systems`
- Review note: technical business system patent description; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `unknown/None/None/None`
- lexical: `unknown/technology/patents/POS_systems`
- minilm: `unknown/technology/patents/POS_systems`
- hybrid: `unknown/technology/patents/POS_systems`

#### FineWeb_000009_028 (FineWeb)

- Text preview: in which the program is operating. If, however, the landmarks have not changed, as indicated by a No at step S131, processing proceeds to step S134. At step S134, the rental information which has been trapped is saved, updated during subsequent interrupt cycles, and ultimately sent to the output buffer at the end of the display. Thereafter, processing returns to step S100 to wait for the next interrupt cycle. If the ...
- Expected: `reference/technology/patents/POS_systems`
- Review note: patent procedure for rental/return data capture; missing_taxonomy: patents/POS_systems; taxonomy_v2_update: software/programming/documentation -> technology/patents/POS_systems
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

### FineMath not expected math (25 records)

#### FineMath_000018_003 (FineMath)

- Text preview: ###### Chapter 10. Experimental Design: Statistical Analysis of ... This is halfway between the 10th and 11th scores. Because both are 0, the median gain is 0. Similarly, the median gain on a running play is 3. The median is a particularly useful measure of central tendency when there are extreme scores at one end of a distribution. Such distributions are said to be skewed in the direction of the extreme scores. The ...
- Expected: `educational/stem/mathematics/statistics`
- Review note: statistics lesson on median/skewed distributions; closest taxonomy math/algebra, missing_taxonomy: statistics; taxonomy_v2_update: stem/mathematics/algebra -> stem/mathematics/statistics
- rule: `unknown/None/None/None`
- lexical: `unknown/stem/mathematics/statistics`
- minilm: `unknown/stem/mathematics/statistics`
- hybrid: `unknown/stem/mathematics/statistics`

#### FineMath_000029_009 (FineMath)

- Text preview: A more serious limitation MAPE occurs when your data set can have 0 values. In the context of sports betting if you’re forecasting winning margins and a draw is possible then you can’t use MAPE. This is because if the final scores are level then Yi = 0 so PEi can’t be calculated due to a division by zero error. For this reason MAPE works best for modeling results such as total basketball, AFL and rugby scores rather ...
- Expected: `educational/stem/mathematics/statistics`
- Review note: forecast error metric with division by zero; missing_taxonomy: statistics; taxonomy_v2_update: stem/mathematics/algebra -> stem/mathematics/statistics
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineMath_000026_012 (FineMath)

- Text preview: 10. Tell students that over the next day or so, they will be able to refine this question based on how they set up their experiment. 11. Prior to showing slide 9 introduce the Tape Pull activity by having the students answer the question in their journal on page 6–4. Slide 9 Student Journal Pages: 6–5 6–6 “You will be working with transparent tape on the tabletop and measuring the force required to remove the tape wi...
- Expected: `educational/science/physics/article`
- Review note: classroom force experiment instructions
- rule: `educational/education/general_education/article`
- lexical: `educational/science/physics/article`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineMath_000026_007 (FineMath)

- Text preview: “We are assuming in this problem that the total force required is equally divided among the six ant feet, and that ONLY the contact between feet and ceiling gives rise to the force.” The weight of the ant is provided in Newtons (N), a derived unit which is the force needed to increase the speed of (or accelerate) one kilogram of mass one meter per second every second. A field test teacher passed around objects (e.g.,...
- Expected: `educational/science/physics/article`
- Review note: force/unit classroom explanation
- rule: `educational/education/general_education/article`
- lexical: `educational/science/physics/article`
- minilm: `unknown/stem/mathematics/arithmetic_measurement`
- hybrid: `educational/stem/mathematics/arithmetic_measurement`

#### FineMath_000005_007 (FineMath)

- Text preview: Solution The problem drew in the figure below. 5. A biker sees the image of a motorcycle behind it 1/6 times its original size when the distance between the biker and motorcycle is 30 meters. Determine the radius of curvature of the rear view mirror… A. 7.14 m B. 8.57 m C. 12.00 m D. 24.00 m Known : Magnification of image (M) = 1/6 times Object distance (d) = 30 meter Wanted: The radius of curvature of the rear view ...
- Expected: `educational/science/physics/article`
- Review note: optics/mirror calculation problem
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

### FineWeb-Edu not expected educational (24 records)

#### FineWeb-Edu_000014_000 (FineWeb-Edu)

- Text preview: Human nature and the nature of war Twenty years ago, Canadian military historian and journalist Gwynne Dyer fascinated the public in 45 countries with his award-winning television series on the nature of war. He followed up that success with an equally remarkable book that grew out of the research that went into the television series. Now Dr. Dyer has published, in his words, "a completely rewritten and updated new e...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: book review / geopolitical reference; missing_taxonomy: history/politics/international_relations
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb-Edu_000018_000 (FineWeb-Edu)

- Text preview: The OED “is widely regarded as the accepted authority on the English language. It is an unsurpassed guide to the meaning, history, and pronunciation of 600,000 words— past and present—from across the English-speaking world. As a historical dictionary, the OED is very different from those of current English, in which the focus is on present-day meanings. You’ll still find these in the OED, but you’ll also find the his...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: library database description for OED; language/reference resource
- rule: `unknown/None/None/None`
- lexical: `unknown/multilingual/mixed_language/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb-Edu_000006_000 (FineWeb-Edu)

- Text preview: Great apes go through mid-life crisis They may not take up surfing or start second careers as cupcake-makers, but chimpanzees and orangutans seem to go through a ‘mid-life crisis’, just like humans. A study of 508 great apes in captivity shows that the animals’ sense of well-being bottoms out in their late 20s to mid-30s, the ape equivalent of middle age, before rebounding in old age. The finding that mid-life crises...
- Expected: `news/science/biology/article`
- Review note: science news about ape well-being study; missing_taxonomy: psychology/animal_behavior
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

#### FineWeb-Edu_000022_002 (FineWeb-Edu)

- Text preview: Picasso, Joan Miró, and Salvador Dalí talked on the radio in his place to publicize the show. Picasso's friend, the surrealist poet Paul Éluard, traveled down to Barcelona from Paris for the exhibit and delivered a lecture at the opening. Éluard, standing in for Picasso, received the hearty support of students who chanted, "Picasso, the Marxist," confirming that they saw Picasso as a radical. (In fact, although Picas...
- Expected: `reference/humanities/art_history/article`
- Review note: historical art/politics narrative; missing_taxonomy: art_history/political_history; taxonomy_v2_update: reference/encyclopedic_article/general -> humanities/art_history/article
- rule: `educational/education/general_education/article`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `educational/education/general_education/article`

#### FineWeb-Edu_000024_000 (FineWeb-Edu)

- Text preview: 1. Yodhājīva Sutta. On five kinds of warriors: those who are frightened by a cloud of dust, by the sight of a flag, by tumult, by conflict, and those who fight victoriously; and on five similar kinds of monks. A.iii.87f. 2. Yodhājīva Sutta. On five kinds of warriors: those who go down into the thick of the fight where they are overpowered, those who are wounded and die on the way to their home, those who survive for ...
- Expected: `reference/reference/encyclopedic_article/general`
- Review note: Buddhist text index/reference entries; missing_taxonomy: religion
- rule: `unknown/None/None/None`
- lexical: `unknown/None/None/None`
- minilm: `unknown/None/None/None`
- hybrid: `unknown/None/None/None`

## Interpretation

- `MiniLM_v2` predicts `source_type=unknown` for all 120 reviewed records. This is expected from the current contract: embedding nearest-label targets semantic taxonomy labels, while source_type is preserved from input or fallback behavior.
- `lexical_v2` predicts `unknown` for 106 / 120 records. It improved semantic labels, but it is not a useful source_type classifier.
- `hybrid_v2` source_type is identical to rule-based because hybrid intentionally takes source_type from rule-based and semantic labels from MiniLM when confidence is high.
- The largest source_type misses are broad real-web categories that current rule keywords do not detect: `reference -> unknown`, `educational -> unknown`, `math -> unknown`, and `reference -> educational`.
- Patent/POS chunks are semantically improved by taxonomy v2, but their source_type remains mostly `unknown` or sometimes `educational`; this suggests source_type needs a type/genre label such as technical_reference or patent_reference if the team wants that distinction.

## Strategy options

### A. Stronger rule-based source_type classifier

Pros: transparent, cheap, no dependencies, easy to tune for obvious web genres such as patent text, product pages, news, forum Q&A, code, and math notation.
Cons: brittle on diverse web text; likely to overfit to benchmark phrases; hard to distinguish educational vs reference vs web_general robustly.
Implementation: add a separate source_type rules module or mode with explicit keyword/pattern tests and source_type-only evaluation.
Overfitting risk: medium to high if rules are written directly from these 120 records.

### B. Separate MiniLM source_type classifier

Pros: aligns with source_type as a genre/format task; can use descriptions for `math`, `reference`, `educational`, `news`, `commercial_product`, `forum_qa`, `web_general`, etc.; avoids mixing source_type with semantic taxonomy.
Cons: needs a separate source_type label set and calibration; may still struggle with subtle boundaries; requires evaluating source_type separately.
Implementation: create source_type taxonomy/descriptions, run nearest-label embeddings over those labels, output `source_type_confidence` and `source_type_method`.
Overfitting risk: medium if descriptions are designed from v1 only; lower if validated on held-out v2-test sample.

### C. Two-stage hybrid

Pros: best conceptual fit: one stage predicts source_type/format, another predicts semantic domain/field/subfield. It matches future probability profiling, where format and topic are different controls.
Cons: more moving parts; requires clear schema and comparison logic; errors can still compound if full-label accuracy is used as the only headline metric.
Implementation: source_type classifier plus semantic classifier, then a merger that keeps separate confidence/method metadata for both tasks.
Overfitting risk: medium; mitigated by freezing benchmark v1 as dev and using a held-out streaming sample for confirmation.

### D. Simplify source_type labels for MVP

Pros: fastest way to make source_type reliable enough for data selection; fewer ambiguous boundaries. Example groups: `math`, `code`, `commercial`, `news_or_reference`, `education_or_explainer`, `forum_qa`, `noise`, `other_web`.
Cons: loses granularity; may be less useful later if NLL/probability profiling needs finer format controls.
Implementation: define a reduced source_type label set, map pseudo-gold labels into it, and reevaluate source_type separately.
Overfitting risk: low to medium, but simplification may hide important distinctions.

## Recommended strategy

Use option C as the target architecture, with option D as a short MVP guardrail. Concretely: keep semantic taxonomy separate, define a compact source_type label set, and test a separate source_type classifier before expanding source_type rules aggressively.
Do not tune MiniLM semantic thresholds to fix source_type. That would optimize the wrong task.

## What not to tune yet

- Do not lower MiniLM semantic threshold to chase full-label accuracy.
- Do not add more semantic taxonomy labels just to improve source_type.
- Do not rewrite rule-based classifier until source_type labels are reviewed and possibly simplified.
- Do not use full-label accuracy as the only headline metric; report source_type and semantic labels separately.

## Relation to future NLL/probability profiling

For observed-token NLL/logprob profiling, `source_type` is likely a format/genre/quality control variable, while `domain/field/subfield` is topical content metadata. These should remain separate because a model may behave differently on patent-like reference text, educational explainers, forum Q&A, product pages, and math derivations even when the semantic domain is similar.
A two-stage labeling design will make later data selection cleaner: filter or stratify by source_type, then compare probability profiles by semantic domain.

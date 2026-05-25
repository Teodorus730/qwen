# Real sample disagreement review

## Purpose

Manual review of rule-based vs lexical labels on the first real HF samples.

## Samples

- FineWeb-Edu: 20 docs, 44 chunks, token_count min/mean/max 83 / 321.07 / 416, agreement 0.5455.

- FineMath: 20 docs, 101 chunks, token_count min/mean/max 81 / 169.99 / 465, agreement 0.4851.

## High-level observations

- FineWeb-Edu has longer chunks and fewer chunks per document.

- FineMath has more chunks and shorter average chunks, often because rows contain many exercises or lesson sections.

- Both real samples have much lower rule-vs-lexical agreement than the synthetic benchmark.

- Full label agreement is a strict signal; disagreement does not automatically mean either label is wrong.

- Real samples are much less clean than the synthetic benchmark and include mixed source/domain cues.

## FineWeb-Edu disagreement types

- Rule-based often assigns broad `education/general_education` while lexical remains low confidence.

- Software licensing and donation/program pages cause disagreement between educational, software documentation, and commercial labels.

- Environmental science/news-like educational text can look like broad education to rules but specific science to lexical matching.

- Commercial/product cues appear in otherwise educational web text.


### FineWeb-Edu_000001_000

- auto-category: domain disagreement, low-confidence lexical

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('educational', None, None, None)` confidence `0.07220698918504387` method `lexical_nearest_label_low_confidence`

- text preview: Taking Play Seriously By ROBIN MARANTZ HENIG Published: February 17, 2008 On a drizzly Tuesday night in late January, 200 people came out to hear a psychiatrist talk rhapsodically about play -- not just the intense, joyous play of children, but play for all people, at all ages, at all times. (All species too; the lecture featured touching photos of a polar bear and a husky engaging playfully at a snowy outpost in northern Canada.) Stuart Brown, president of the National Institute for Play, was speaking at the New York Public Library's main branch on 42nd Street. He created the institute in 1996, after more than 20 years of psychiatric practice and research persuaded him of the dangerous long


### FineWeb-Edu_000001_001

- auto-category: domain disagreement, low-confidence lexical

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('educational', None, None, None)` confidence `0.05765272462919676` method `lexical_nearest_label_low_confidence`

- text preview: floor or climb trees in the woods, today's children are missing out on something essential. The success of ''The Dangerous Book for Boys'' -- which has been on the best-seller list for the last nine months -- and its step-by-step instructions for activities like folding paper airplanes is testament to the generalized longing for play's good old days. So were the questions after Stuart Brown's library talk; one woman asked how her children will learn trust, empathy and social skills when their most frequent playing is done online. Brown told her that while video games do have some play value, a true sense of ''interpersonal nuance'' can be achieved only by a child who is engaging all five sen


### FineWeb-Edu_000001_002

- auto-category: domain disagreement, low-confidence lexical

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('educational', None, None, None)` confidence `0.05738414360061109` method `lexical_nearest_label_low_confidence`

- text preview: a little too eager -- to promote a scientific argument for play. They have spent the past few decades learning how and why play evolved in animals, generating insights that can inform our understanding of its evolution in humans too. They are studying, from an evolutionary perspective, to what extent play is a luxury that can be dispensed with when there are too many other competing claims on the growing brain, and to what extent it is central to how that brain grows in the first place. Scientists who study play, in animals and humans alike, are developing a consensus view that play is something more than a way for restless kids to work off steam; more than a way for chubby kids to burn off 


### FineWeb-Edu_000003_000

- auto-category: domain disagreement, rule_based_unknown vs lexical label, boilerplate/noise

- rule_based: `('educational', None, None, None)` confidence `0.2` method `rule_based_unknown`

- lexical: `('educational', 'software', 'programming', 'documentation')` confidence `0.08770580193070293` method `lexical_nearest_label_v1`

- text preview: CTComms sends on average 2 million emails monthly on behalf of over 125 different charities and not for profits. Take the complexity of technology and stir in the complexity of the legal system and what do you get? Software licenses! If you've ever attempted to read one you know how true this is, but you have to know a little about software licensing even if you can't parse all of the fine print. By: Chris Peters March 10, 2009 A software license is an agreement between you and the owner of a program which lets you perform certain activities which would otherwise constitute an infringement under copyright law. The software license usually answers questions such as: The price of the software 


### FineWeb-Edu_000003_001

- auto-category: domain disagreement, rule_based_unknown vs lexical label, boilerplate/noise

- rule_based: `('educational', None, None, None)` confidence `0.2` method `rule_based_unknown`

- lexical: `('educational', 'software', 'programming', 'documentation')` confidence `0.11870660330711032` method `lexical_nearest_label_v1`

- text preview: software and the supporters of open-source software agree with one another on most issues. However, the official definition of free software differs somewhat from the official definition of open-source software, and the philosophies underlying those definitions differ as well. For a short description of the difference, read Live and Let License. For a longer discussion from the "free software" side, read Why Open Source Misses the Point of Free Software. For the "open-source" perspective, read Why Free Software is Too Ambiguous. Public domain and copyleft. These terms refer to different categories of free, unrestricted licensing. A copyleft license allows you all the freedoms of a free softw


### FineWeb-Edu_000003_002

- auto-category: source_type disagreement, domain disagreement, boilerplate/noise, commercial false positive candidate

- rule_based: `('commercial_product', 'commercial', 'product_page', 'retail')` confidence `0.82` method `rule_based_keyword_v1`

- lexical: `('educational', 'software', 'programming', 'documentation')` confidence `0.0818926635648921` method `lexical_nearest_label_v1`

- text preview: continue using it. End User Licensing Agreement (EULA). When you acquire software yourself, directly from a vendor or retailer, or directly from the vendor's Web site, you usually have to indicate by clicking a box that you accept the licensing terms. This "click-through" agreement that no one ever reads is commonly known as a EULA. If you negotiate a large purchase of software with a company, and you sign a contract to seal the agreement, that contract usually replaces or supersedes the EULA. Most major vendors of proprietary software offer some type of bulk purchasing and volume licensing mechanism. The terms vary widely, but if you order enough software to qualify, the benefits in terms o


### FineWeb-Edu_000003_003

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, boilerplate/noise, commercial false positive candidate

- rule_based: `('commercial_product', 'commercial', 'product_page', 'retail')` confidence `0.82` method `rule_based_keyword_v1`

- lexical: `('educational', None, None, None)` confidence `0.07919765405964414` method `lexical_nearest_label_low_confidence`

- text preview: Software Donation Program FAQ. For general information about the volume licensing of Microsoft software, see Volume Licensing Overview. If you get Microsoft software from TechSoup or other software distributors who work with not-for-profits, you may need to go to the eOpen Web site to locate your Volume license keys. For more information, check out the TechSoup Donation Recipient's Guide to the Microsoft eOpen Web Site. Always check TechSoup Stock first to see if there's a volume licensing donation program for the software you're interested in. If TechSoup doesn't offer that product or if you need more copies than you can find at TechSoup, search for "volume licensing not-for-profits softwar


### FineWeb-Edu_000003_004

- auto-category: domain disagreement, low-confidence lexical, boilerplate/noise

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('educational', None, None, None)` confidence `0.04876491113537703` method `lexical_nearest_label_low_confidence`

- text preview: words, while a User CAL (Client Access License) for Windows Server is distinct from a User CAL for SharePoint Server, the underlying terms and rights are very similar. The TechSoup product pages for Microsoft software do a good job of describing the differences between products, so we'll focus on the common threads in this article. Moreover, Microsoft often lets you license a single server application in more than one way, depending on the needs of your organisation. This allows you the flexibility to choose the licenses that best reflect your organisation's usage patterns and thereby cost you the least amount of money. For example, for Windows Server and other products you can acquire licen


## FineMath disagreement types

- Math source text frequently uses lesson/curriculum wording, so rule-based labels it as education while lexical preserves math source_type.

- Rule-based often leaves math domain null when formulas are present but the exact algebra/calculus cues are weak.

- Lexical sometimes assigns `stem/mathematics/algebra` or `calculus` from keyword overlap where rule-based remains unknown.

- Web noise, product/resource listings, and cookie text appear inside math-sourced rows.


### FineMath_000001_000

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.024847756235032988` method `lexical_nearest_label_low_confidence`

- text preview: Our curriculum is spiral Please note that our virtual Singapore Math Grade 3 curriculum is spiral and it provides for the review of the important concepts that students learned in Grade 2. Our online K-5 math curriculum is aligned with all standard Singapore Math textbook series and it includes all content that these series cover, from Kindergarten grade through 5th grade. Our Singapore Math for 3rd Grade may introduce some topics one grade level earlier or postpone coverage of some topics until grade 4. In the few instances where 3rd grade level units don’t exactly align between our curriculum and textbooks, you will still be able to easily locate the corresponding unit in our program by re


### FineMath_000001_001

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.039103094350288754` method `lexical_nearest_label_low_confidence`

- text preview: Multiplication and division, multiplication tables of 2, 5, and 10, multiplication tables of 3 and 4, multiplication tables of 6, 7, 8, and 9, solving problems involving multiplication and division, and mental math computation and estimation. Singapore Math 3b Understanding fractions, time, volume, mass, representing and interpreting data, area and perimeter, and attributes of two-dimensional shapes. Student prior knowledge Prior to starting third grade Singapore Math, students should already know how to relate three-digit numbers to place value, use place-value charts to form a number and compare three-digit numbers. The initial lessons in the Singapore Math 3rd Grade are both a review and 


### FineMath_000001_002

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.07519061581466271` method `lexical_nearest_label_low_confidence`

- text preview: This unit covers understanding of multiplication and division. In this unit students will extend their knowledge of making equal groups to formalize their understanding of multiplication and division. The focus of this unit is on understanding multiplication and division using equal groups, not on memorizing facts. Students will learn how the multiplication symbol to represent addition of quantities in groups. • Multiplication Tables Of 2, 5, And 10 This unit covers multiplying by 2 using skip-counting, multiplying by 2 using dot paper, multiplying by 5 using skip-counting, multiplying by 5 using dot paper, multiplying by 10 using skip-counting, dividing using related multiplication facts of


### FineMath_000001_003

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.07027283689263066` method `lexical_nearest_label_low_confidence`

- text preview: • Multiplication Tables Of 3 And 4 This unit covers multiplying by 3 using skip-counting, multiplying by 3 using dot paper, multiplying by 4 using skip-counting, multiplying by 4 using dot paper, dividing using related multiplication facts of 3 or 4. Students will learn building multiplication tables of 3 and 4 to formalize their understanding of multiplication and division for facts 3 and 14. Students will learn to find their division facts by thinking of corresponding multiplication facts. • Multiplication Tables Of 6, 7, 8, And 9


### FineMath_000001_004

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.06888365903182905` method `lexical_nearest_label_low_confidence`

- text preview: This unit covers multiplication properties, multiplying by 6, multiplying by 7, multiplying by 8, multiplying by 9, dividing using related multiplication facts of 6, 7, 8, or 9. Students will learn building multiplication tables of 6, 7, 8, and 9 to formalize their understanding of multiplication and division for facts 6, 7, 8, and 9. Students will learn to find their division facts by thinking of corresponding multiplication facts. • Solving Problems Involving Multiplication And Division This unit covers solving one- and two-step word problems involving multiplication and division. Students will use a part-whole and comparison models to solve word problems involving multiplication and divis


### FineMath_000001_005

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education, boilerplate/noise

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.06077371254627639` method `lexical_nearest_label_low_confidence`

- text preview: This unit covers parts and wholes, fractions and number lines, comparing unit fractions, equivalence of fractions, and comparing like fractions. Students will learn fractional notation that include the terms “numerator” and “denominator.” Students will understand that a common fraction is composed of unit fractions and they will learn to compare unit fractions. • Time This unit covers telling time, adding time, subtracting time, and time intervals. Students will review and practice to tell time to the minute, learn telling intervals of time in hours, convert units of time between hours, minutes, seconds, days and weeks. • Volume This unit covers understanding of volume, comparing volume, mea


### FineMath_000001_006

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.05482930791331408` method `lexical_nearest_label_low_confidence`

- text preview: This unit covers scaled picture graphs and scaled bar graphs, reading and interpreting bar graphs, and line plots. Students will learn to sort data into groups and categories and use numerical data to interpret bar graphs and line plots. • Area And Perimeter This unit covers understanding of area, measuring area using square centimeters and square inches, measuring area using square meters and square feet, area and perimeter, solving problems involving area and perimeter. Students will learn to find and measure area of figures in square units that include square centimeters, square inches, square meters and square feet. • Attributes Of Two-Dimensional Shapes This unit covers categories and a


### FineMath_000002_000

- auto-category: source_type disagreement, domain disagreement, math-vs-education, boilerplate/noise, commercial false positive candidate

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', 'commercial', 'product_page', 'retail')` confidence `0.08299250027587321` method `lexical_nearest_label_v1`

- text preview: # Fractions Bundle "Twist" 12 Worksheets Subject Resource Type Product Rating File Type Word Document File 181 KB|20 pages Share Product Description FRACTIONS BUNDLE "TWIST" 12 WORKSHEETS You receive 12 FULL worksheets all on the 4 OPERATIONS of FRACTIONS including WORD PROBLEMS for EACH operation! The "Twist"? - ALL these examples are written out IN WORDS to make your students THINK a little more. They must first READ the fractions correctly, then WRITE them correctly and then SOLVE them correctly (hopefully!) to lowest terms; Worksheet 1 - ADDITION OF FRACTIONS; all types of addition of fraction examples with like and unlike denominators; 15 examples plus a BONUS at the end; Worksheet 2 - 


### FineMath_000002_001

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education, boilerplate/noise

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.042893574097432455` method `lexical_nearest_label_low_confidence`

- text preview: Worksheet 4 - DIVISION OF FRACTIONS; these 15 examples also involve cancelling when possible PLUS a GEOMETRY BONUS question; Worksheets 5 and 6 - MULTIPLICATION AND DIVISION OF FRACTIONS COMBINED; students must multiply or divide these examples using cancelling; 21 examples plus many BONUS fraction questions included; Worksheets 7 and 8 - ALL OPERATIONS PLUS!; these TWO worksheets combine ALL 4 Operations; 16 examples, 4 for each operation included; PLUS 8 BONUS questions on fractions; Worksheets 9 and 10 - WORD PROBLEMS for ADDITION & SUBTRACTION OF FRACTIONS; NO FLUFF! Just 25 word problems all on addition and/or subtraction of fractions; students must solve not only for the correct operat


### FineMath_000002_002

- auto-category: source_type disagreement, domain disagreement, low-confidence lexical, math-vs-education, boilerplate/noise

- rule_based: `('educational', 'education', 'general_education', 'article')` confidence `0.62` method `rule_based_keyword_v1`

- lexical: `('math', None, None, None)` confidence `0.04879500364742666` method `lexical_nearest_label_low_confidence`

- text preview: The thumbnails only give you 4 items to look at. The download preview gives you more. Take a look! * ALSO AVAILABLE FOR YOU OR A COLLEAGUE! - CLICK ANY LINK YOU WANT: - FRACTIONS REVIEW AND REINFORCEMENT - 10 FULL worksheets on various aspects of fractions to be used as great review or part of your unit. A LOT of good work here for your students. This link will describe all 10 pages in detail. - FRACTIONS POWERPOINT FUN QUIZ - 60 slides in all! This Powerpoint program starts over again EACH time students get even one answer wrong! Challenging, fun and SELF-CORRECTING! Great activity with great graphics will keep students engaged. NOT for the beginner! Did I say it was self-correcting?


## Interpretation

- Full labels are too strict as a single success metric for real samples.

- Null `domain/field/subfield` can be acceptable for low-confidence real chunks.

- Disagreement is useful as a triage signal for manual review and future taxonomy decisions.

- Dataset/source_type and inferred domain are separate layers; FineMath source text can still be educational, commercial, noisy, or formula-heavy.

## Implications for next steps

- Do not blindly optimize rule-based labels to match lexical labels.

- Improve taxonomy/rules only after manual decisions on representative disagreements.

- The first NLL pilot should include both stable-label chunks and disagreement chunks.

- Build a small human review set before treating real-sample labels as benchmark truth.

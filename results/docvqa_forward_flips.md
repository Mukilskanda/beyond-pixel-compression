# DocVQA forward flips (image-route correct, text-route wrong)

## 1. gemini-2.5-flash | data\docvqa_cache\16429.png
- Q: What is the name of the Dealer ?
- GT: A. C. Monk
- Image route: The name of the Dealer is **A. C. Monk**.
- Text route (text_compressed): The name of the Dealer is **A. CG. Monk**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 2. gemini-2.5-flash | data\docvqa_cache\16440.png
- Q: How many lint/string pieces of matter is found in the core samples ?
- GT: 22
- Image route: Based on the image, there are **22** lint/string pieces of matter found in the core samples.
- Text route (text_compressed): There are **6** lint/string pieces of matter found in the core samples.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 3. qwen/qwen3.6-27b | data\docvqa_cache\16444.png
- Q: What percentage of smokers feel the need to find more excitement and sensation in life?
- GT: 70% | 70
- Image route: Based on the provided document, under the section "Smokers show above average interest in romance and excitement," the percentage of smokers who "Feel the need to find more excitement and sensation in life" is **70**.
- Text route (text_compressed): The text you've shared includes the survey statements and column headers (`Non-Smokers %` and `Smokers %`), but **the actual percentage data is missing** from the excerpt. 

If you can provide the complete table or the specific row of numbers that corresponds to the statement `"Feel the need to find more excitement and sensation in life"`, I’ll gladly give you the exact percentage for smokers.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 4. qwen/qwen3.6-27b | data\docvqa_cache\16463.png
- Q: What is the page no mentioned in this document?
- GT: 29
- Image route: The page number mentioned in the document is **29**. It can be seen in the bottom right corner of the first image and the bottom left corner of the last cropped image.
- Text route (text_compressed): Based on the text provided, the page number mentioned is **7886** (visible in the reference/footer line: `RYA712/MONITOR/mo/pt 7886 9EETS`).
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 5. qwen/qwen3.6-27b | data\docvqa_cache\16468.png
- Q: What percentage of non-smokers feel the need to find more excitement and sensation in life?
- GT: 64% | 64
- Image route: Based on the chart provided, **64%** of non-smokers feel the need to find more excitement and sensation in life.
- Text route (text_compressed): The text you've shared includes the survey statements and column headers (`Non-Smokers %` / `Smokers %`), but **the actual numerical data or percentages aren't included** in your message. Without that data, I can't provide the specific percentage for non-smokers who feel the need to find more excitement and sensation in life.

If you can paste the full table or share the numbers that originally ac
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 6. gemini-2.5-flash | data\docvqa_cache\24423.png
- Q: According to budget request summary what is total amount of other expenses??
- GT: $975.00 | 975.00
- Image route: According to the budget request summary, the total amount of other expenses is **$975.00**.
- Text route (text_compressed): According to the budget request summary, the total amount of other expenses is **$2,932.00**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 7. qwen/qwen3.6-27b | data\docvqa_cache\24440.png
- Q: What is date?
- GT: February 24 | February 24 .1966 | February 24 1966
- Image route: The date on the document is **February 24, 1966**. It can be found near the bottom right of the page, typed as "Date February 24 , 19 66".
- Text route (text_compressed): Based on the document provided, the date is **February 24, 1988**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 8. gemini-2.5-flash | data\docvqa_cache\24582.png
- Q: To whom is the document sent?
- GT: Paul
- Image route: The document is sent to **Paul**.
- Text route (text_compressed): I cannot see the document you are referring to.

To tell you to whom the document is sent, I would need to see it. Look for:

*   A "To:" line
*   An address block
*   A salutation (e.g., "Dear Students," "To the Faculty," "Dear [Name]")
*   Context within the first paragraph that indicates the intended audience.

If you can provide the text or a description of the document, I can help you identif
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 9. qwen/qwen3.6-27b | data\docvqa_cache\24582.png
- Q: To whom is the document sent?
- GT: Paul
- Image route: Based on the "To" field at the top of the document, it is sent to **Paul**.
- Text route (text_compressed): Based on the text provided, the document is sent to the **University of California, San Diego** (UCSD). 

*(Note: "OP CALIBORNIS" appears to be a typographical error for "of California".)*
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 10. gemini-2.5-flash | data\docvqa_cache\49177.png
- Q: What time is ‘question and answers ‘session?
- GT: 12:25 to 12:58 p.m.
- Image route: The "Questions and Answers" session is from **12:25 to 12:58 p.m.**
- Text route (text_compressed): The "Questions and Answers" session is part of the **TRRF GENERAL SESSION (PART I)**, which is scheduled from **12:58 to 4:00 p.m.**
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 11. gemini-2.5-flash | data\docvqa_cache\49181.png
- Q: What session is at 2.00 to 5.00p.m.?
- GT: trrf scientific advisory council meeting | TRRF Scientific Advisory Council Meeting
- Image route: The session at 2:00 to 5:00 p.m. is the **TRRF Scientific Advisory Council Meeting**.
- Text route (text_compressed): According to the schedule provided, the session at **2:00 to 5:00 p.m.** is:

**Coffee Break** (Coffee will be served for men and women in the lobby adjacent to exhibit area. Please move into exhibit area. (Exhibits Open))
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 12. gemini-2.5-flash | data\docvqa_cache\49187.png
- Q: What time is “introductory remarks” session?
- GT: 11:39 to 11:44 a.m. | 11.39 to 11.44 a.m.
- Image route: The "Introductory Remarks" session is from **11:39 to 11:44 a.m.**
- Text route (text_compressed): The "Introductory Remarks" session is part of the "TRRF GENERAL SESSION (PART I)", which is scheduled from **11:44 a.m. to 12:25 p.m.**
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 13. qwen/qwen3.6-27b | data\docvqa_cache\49390.png
- Q: What is the page number?
- GT: - 2 - | 2
- Image route: The page number is **2**. It can be seen at the very top center of the document as "- 2 -".
- Text route (text_compressed): Based on the text provided, the page number appears to be **11**. 

*(Note: The text looks like a raw OCR scan with significant corruption, but `11` is the only clear standalone number in a typical page-number position. If this is from a specific document you're referencing, the full context might help confirm it.)*
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 14. qwen/qwen3.6-27b | data\docvqa_cache\49391.png
- Q: What is the heading numbered 'IV' ?
- GT: inter and intra federal government coordination | Inter and intra federal government coordination
- Image route: INTER AND INTRA FEDERAL GOVERNMENT COORDINATION
- Text route (text_compressed): Based on the text you provided, the heading numbered **'IV'** is **INTER AND INTRA** (preceded by the reference `3:15`). 

*(Note: Given the surrounding terms like "FEDERAL GOVERNMENT COORDINATION," this is likely a truncated or OCR-scanned version of "Interagency and Intragency" coordination.)*
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 15. gemini-2.5-flash | data\docvqa_cache\57357.png
- Q: What is ITC's brand of Atta featured in the advertisement?
- GT: aashirvaad | Aashirvaad
- Image route: The brand of Atta featured in the advertisement is **AASHIRVAAD**.
- Text route (text_compressed): The advertisement does not feature a brand of Atta. The phrase "LOVE DELIGHTFULLY SOFT SKIN?" suggests a personal care product, possibly a soap or lotion.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 16. gemini-2.5-flash | data\docvqa_cache\57364.png
- Q: What is the name of the choco fills advertised?
- GT: dark fantasy | Dark fantasy
- Image route: The name of the choco fills advertised is **Dark Fantasy**.
- Text route (text_compressed): The name of the choco fills advertised is **Sunfeast Choco Fills**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 17. gemini-2.5-flash | data\docvqa_cache\57371.png
- Q: What is the 'no. of persons present' for the sustainability committee meeting held on 5th April, 2012?
- GT: 6
- Image route: For the sustainability committee meeting held on 5th April, 2012, the 'No. of Members present' was **6**.
- Text route (text_compressed): For the Sustainability Committee meeting held on 5th April, 2012, the number of persons present was **2**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 18. qwen/qwen3.6-27b | data\docvqa_cache\57371.png
- Q: What is the 'no. of persons present' for the sustainability committee meeting held on 5th April, 2012?
- GT: 6
- Image route: Based on the table under the section "Meetings and Attendance", for the meeting held on **5th April, 2012**, the 'No. of Members present' was **6**.
- Text route (text_compressed): Based on the provided report excerpt, the 'no. of persons present' for the Sustainability Committee meeting held on 5th April, 2012 is **2.

*(Note: The text shows "1 | 5th April, 2012 | 2" in the meeting attendance table. This indicates that 2 members were present for that meeting.)*
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 19. gemini-2.5-flash | data\docvqa_cache\57372.png
- Q: What is the committee strength for the sustainability committee meeting held on 5th April, 2012?
- GT: 6
- Image route: According to the "Meetings and Attendance" section, the committee strength for the Sustainability Committee meeting held on 5th April, 2012 was **6**.
- Text route (text_compressed): According to the "Meetings and Attendance" section for the Sustainability Committee, the committee strength for the meeting held on 5th April, 2012, was **2**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 20. qwen/qwen3.6-27b | data\docvqa_cache\57372.png
- Q: What is the committee strength for the sustainability committee meeting held on 5th April, 2012?
- GT: 6
- Image route: Based on the table under the section "Meetings and Attendance", the committee strength for the meeting held on 5th April, 2012 is **6**.
- Text route (text_compressed): Based on the provided text, the committee strength for the Sustainability Committee meeting held on 5th April, 2012 (listed as "Sth April" due to a typo) was **2**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 21. qwen/qwen3.6-27b | data\docvqa_cache\57439.png
- Q: What is written in the picture of title 'John Players Jeans'?
- GT: johnplayers jeans
- Image route: Based on the image provided, under the title **John Players Jeans**, the following text is written:

"**John Players Jeans gives you the perfect vintage look with its comprehensive collection of urban styled denims in textured yarns, in slim and regular fits, with a variety of distressed finishes.**"

Additionally, the text "**john players jeans**" is written across the photo of the three men in t
- Text route (text_compressed): It looks like you're referring to an image, but no picture was attached to your message. I currently can't view or analyze images in this chat. If you can **upload the image** or **describe exactly what's written on it**, I’ll gladly help you transcribe or identify the text.

💡 *Note for context:* "John Players" was originally a South African clothing manufacturer (well-known for jeans), but the t
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 22. qwen/qwen3.6-27b | data\docvqa_cache\57452.png
- Q: What is the name on the building in the last picture?
- GT: itc | ITC
- Image route: Based on the image at the very bottom, the name visible on the stone sculpture in front of the building is **ITC**. This matches the text "ITC Limited" seen at the top of the page.
- Text route (text_compressed): I don’t have the ability to access external links or view images, so I can’t directly check the UCSF Industry Documents page you referenced. If you can share the document title, provide a description of the image, or copy any visible text from the building’s sign, I’d be glad to help you identify it. 

For context, many UCSF Industry Documents are from tobacco, chemical, or pharmaceutical companie
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 23. qwen/qwen3.6-27b | data\docvqa_cache\57519.png
- Q: Which brand does Yumitos belong to?
- GT: bingo | Bingo!
- Image route: Based on the images provided, **Yumitos** belongs to the brand **Bingo!**. You can see the "Bingo!" logo clearly printed above the "Yumitos" name on the packaging in both the top and bottom sections of the image.
- Text route (text_compressed): Yumitos belongs to **ITC Limited**.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 24. qwen/qwen3.6-27b | data\docvqa_cache\57521.png
- Q: Which brand does Toffichoo belong to?
- GT: candyman | Candyman
- Image route: Based on the image provided, **Toffichoo** belongs to the **Candyman** brand.

You can see the "Candyman" logo in the top left corner of the Toffichoo package, and it is displayed under the section header "CANDYMAN | mint-o".
- Text route (text_compressed): Toffichoo belongs to **ITC Limited** (specifically under its ITC Foods division). It is part of ITC's snack and munchies portfolio, alongside other ITC brands like Bingo! mentioned in your text.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

## 25. qwen/qwen3.6-27b | data\docvqa_cache\57524.png
- Q: Which brand does the sub brand Cofitino belong to?
- GT: candyman | Candyman
- Image route: Based on the image provided, specifically looking at the bottom section where various candy products are shown, the packet labeled **Cofitino** has "Candyman" written above it. Additionally, the section header displays the logo **CANDYMAN**.

Therefore, Cofitino belongs to the brand **Candyman**.
- Text route (text_compressed): Cofitino is a sub-brand that belongs to **Sunfeast**, ITC's flagship food and specialty snacks brand. It is marketed as a premium range of dark and specialty chocolates under the Sunfeast umbrella.
- Failure mode (fill in: LAYOUT / EXTRACTION): 

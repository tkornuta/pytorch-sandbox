# SPDX-License-Identifier: Apache-2.0
# based on: https://medium.com/@shahrukhx01/a-novel-approach-for-text-to-sql-dual-transformers-approach-e2a285dfb630

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

## define the model hub's model name
model_name = "shahrukhx01/schema-aware-denoising-bart-large-cnn-text2sql"

## load model and tokenizer
model = BartForConditionalGeneration.from_pretrained('shahrukhx01/schema-aware-denoising-bart-large-cnn-text2sql')
tokenizer = BartTokenizer.from_pretrained('shahrukhx01/schema-aware-denoising-bart-large-cnn-text2sql')

# prepare question, this is how the table header looks like for this example
##['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team']

## we have to encode schema and concat the question alonside it as follows
question_schema = """What is justin's  nationality? """
                      #</s> <col0> Player : text <col1> No. : text <col2> Nationality : text 
                      #<col3> Position : text <col4> Years in Toronto : text <col5>  School/Club Team : text"""

## tokenize question_schema
inputs = tokenizer([question_schema], max_length=1024, return_tensors='pt')

# generate SQL
text_query_ids = model.generate(inputs['input_ids'], num_beams=4, min_length=0, max_length=125, early_stopping=True)
prediction = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in text_query_ids][0]

##magic!
print(prediction)
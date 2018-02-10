from Topic.post_SMS.ML_from_scratch import GetFeatures
from Topic.post_SMS.ML_from_scratch import Model
from Topic.post_SMS.ML_from_scratch import Config
from datetime import datetime

'''
The next line reads in raw messages from a csv and transforms it into a feature file. 
'''
GetFeatures.Main()

'''
The next block of text reads a pre-existing features file, trains and tests models for each topic, then writes a file with performance scores
'''
model = Model.Main()

scores = model.models
topics = sorted(scores)

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
results_file = Config.output_dir + Config.classifier + '_' + str(now) + '.txt'

with open(results_file, 'w') as t:
    t.write("Results generated: " + str(now))
    t.write('\n')
    t.write("Platform: " + Config.platform)
    t.write('\n')
    t.write("Classifier: " + Config.classifier)
    t.write('\n')
    t.write("Folds: " + str(Config.k_folds))
    t.write('\n')
    t.write("Features:" + str(model.feature_file))
    t.write('\n')
    t.write('\t' + 'Unigrams: ' + str(Config.num_unigrams))
    t.write('\n')
    t.write('\t' + 'Bigrams: ' + str(Config.num_bigrams))
    t.write('\n')
    t.write('\t' + 'POS tags: ' + str(Config.pos_restriction))
    t.write('\n')
    t.write('\t' + 'Mentions: ' + str(Config.use_mentions))
    t.write('\n\n')
    for f in topics:
        t.write(f)
        t.write('\n')
        t.write(scores[f])
        t.write('\n')


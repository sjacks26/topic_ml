import GetFeatures
import Model


'''
The next line reads in raw messages from a csv and transforms it into a feature file. 
'''
GetFeatures.main()

'''
The next block of text reads a pre-existing features file, trains and tests models for each topic, then writes a file with performance scores
'''
Model.main()

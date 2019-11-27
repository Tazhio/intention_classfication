import argparse
import sys
from INTENTS.Instantiation import Instantiation
# from INTENTS.Delete import
from combined_model import intent_classifier



if __name__ == '__main__':
    user_sentence = sys.argv[1]
    intent=intent_classifier(user_sentence)
    if(intent==1):
        instant = Instantiation()
        instant.analyze(user_sentence)





    # network_file = args.network_file
    # seed = args.seed
    # diffusion_model = args.diffusion_model
    # time_limit = args.time_limit
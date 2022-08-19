"""bla"""
# from __future__ import print_function
import copy
import sklearn
import numpy as np
import pandas as pd
from . import limes
from .limes import lime_tabular

# import string
import os
os.environ['SPACY_WARNING_IGNORE'] = 'W008'
import sys
sys.path.append(os.getcwd()+'/dataset/mortality')

if (sys.version_info > (3, 0)):
    def unicode(s, errors=None):
        return s#str(s)

class Bunch(object):
    """bla"""
    def __init__(self, adict):
        self.__dict__.update(adict)


def map_array_values(array, value_map):
    # value map must be { src : target }
    ret = array.copy()
    for src, target in value_map.items():
        ret[ret == src] = target
    return ret
def replace_binary_values(array, values):
    return map_array_values(array, {'0': values[0], '1': values[1]})

def load_dataset(dataset_name, balance=False, discretize=True, dataset_folder='./', X=None, y=None, plot=False):


    if plot or "generate" in dataset_name or "artificial" in dataset_name:
        if "blobs" in dataset_name:
            alphabet = ["a", "b", "c", "d", "e", "f","g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
            feature_names = alphabet[:len(X[0])]
            feature_names.append("class")
        else:
            feature_names = ["x", "y", "class"]
        
        dataset = load_csv_dataset(
            np.column_stack((X,y)), -1, ', ',
            feature_names=feature_names, discretize=discretize, balance=balance, data_generate=True)
        dataset.class_names = ['class ' + str(Y)  for Y in range(len(y))]

    elif "iris" in dataset_name:
        feature_names = ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width", "class"]
        dataset = load_csv_dataset(
            np.column_stack((X,y)), -1, ', ',
            feature_names=feature_names, discretize=discretize, balance=balance, data_generate=True)
        dataset.class_names = ['Setosa', 'Versicolour', 'Virginica']

    elif dataset_name == 'adult':
        feature_names = ["Age", "Workclass", "fnlwgt", "Education",
                         "Education-Num", "Marital Status", "Occupation",
                         "Relationship", "Race", "Sex", "Capital Gain",
                         "Capital Loss", "Hours per week", "Country", 'Income']
        features_to_use = [0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        categorical_features = [1, 3, 5, 6, 7, 8, 9, 10, 11, 13]
        education_map = {
            '10th': 'Dropout', '11th': 'Dropout', '12th': 'Dropout', '1st-4th':
            'Dropout', '5th-6th': 'Dropout', '7th-8th': 'Dropout', '9th':
            'Dropout', 'Preschool': 'Dropout', 'HS-grad': 'High School grad',
            'Some-college': 'High School grad', 'Masters': 'Masters',
            'Prof-school': 'Prof-School', 'Assoc-acdm': 'Associates',
            'Assoc-voc': 'Associates',
        }
        occupation_map = {
            "Adm-clerical": "Admin", "Armed-Forces": "Military",
            "Craft-repair": "Blue-Collar", "Exec-managerial": "White-Collar",
            "Farming-fishing": "Blue-Collar", "Handlers-cleaners":
            "Blue-Collar", "Machine-op-inspct": "Blue-Collar", "Other-service":
            "Service", "Priv-house-serv": "Service", "Prof-specialty":
            "Professional", "Protective-serv": "Other", "Sales":
            "Sales", "Tech-support": "Other", "Transport-moving":
            "Blue-Collar",
        }
        country_map = {
            'Cambodia': 'SE-Asia', 'Canada': 'British-Commonwealth', 'China':
            'China', 'Columbia': 'South-America', 'Cuba': 'Other',
            'Dominican-Republic': 'Latin-America', 'Ecuador': 'South-America',
            'El-Salvador': 'South-America', 'England': 'British-Commonwealth',
            'France': 'Euro_1', 'Germany': 'Euro_1', 'Greece': 'Euro_2',
            'Guatemala': 'Latin-America', 'Haiti': 'Latin-America',
            'Holand-Netherlands': 'Euro_1', 'Honduras': 'Latin-America',
            'Hong': 'China', 'Hungary': 'Euro_2', 'India':
            'British-Commonwealth', 'Iran': 'Other', 'Ireland':
            'British-Commonwealth', 'Italy': 'Euro_1', 'Jamaica':
            'Latin-America', 'Japan': 'Other', 'Laos': 'SE-Asia', 'Mexico':
            'Latin-America', 'Nicaragua': 'Latin-America',
            'Outlying-US(Guam-USVI-etc)': 'Latin-America', 'Peru':
            'South-America', 'Philippines': 'SE-Asia', 'Poland': 'Euro_2',
            'Portugal': 'Euro_2', 'Puerto-Rico': 'Latin-America', 'Scotland':
            'British-Commonwealth', 'South': 'Euro_2', 'Taiwan': 'China',
            'Thailand': 'SE-Asia', 'Trinadad&Tobago': 'Latin-America',
            'United-States': 'United-States', 'Vietnam': 'SE-Asia'
        }
        married_map = {
            'Never-married': 'Never-Married', 'Married-AF-spouse': 'Married',
            'Married-civ-spouse': 'Married', 'Married-spouse-absent':
            'Separated', 'Separated': 'Separated', 'Divorced':
            'Separated', 'Widowed': 'Widowed'
        }
        label_map = {'<=50K': 'Less than $50,000', '>50K': 'More than $50,000'}

        def cap_gains_fn(x):
            x = x.astype(float)
            d = np.digitize(x, [0, np.median(x[x > 0]), float('inf')],
                            right=True).astype('|S128')
            return map_array_values(d, {'0': 'None', '1': 'Low', '2': 'High'})

        transformations = {
            3: lambda x: map_array_values(x, education_map),
            5: lambda x: map_array_values(x, married_map),
            6: lambda x: map_array_values(x, occupation_map),
            10: cap_gains_fn,
            11: cap_gains_fn,
            13: lambda x: map_array_values(x, country_map),
            14: lambda x: map_array_values(x, label_map),
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'adult/adult.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, discretize=discretize,
            balance=balance, feature_transformations=transformations)
        dataset.transformations = transformations

    elif dataset_name == 'titanic':
        feature_names = ["PassengerId", "Pclass",  "First Name", "Last Name", "Sex",
                "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Survived"]
        features_to_use = [1, 4, 5, 6, 7, 11]
        categorical_features = [1, 4, 6, 7, 11]
        
        sex_map = {0: 'Female', 1: 'Male'}
        pclass_map = {1: '1st', 2: '2nd', 3: '3rd'}
        transformations = {
            4: lambda x: map_array_values(x, sex_map),
            1: lambda x: map_array_values(x, pclass_map)
        }
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'titanic/titanic.data'), -1, ', ',
            feature_names=feature_names, features_to_use=features_to_use,
            categorical_features=categorical_features, 
            discretize=discretize,
            balance=balance, feature_transformations=transformations)
        dataset.class_names = ['Survived', 'Died']
        dataset.transformations = transformations
        
    elif dataset_name == 'compas':
        """
        feature_names = ['id', 'name', 'first', 'last', 'compas_screening_date', 'sex', 'dob', 
                        'age', 'age_cat', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 
                        'juv_other_count', 'priors_count', 'days_b_screening_arrest', 'c_jail_in',
                        'c_jail_out', 'c_case_number', 'c_offense_date', 'c_arrest_date', 'c_days_from_compas',
                        'c_charge_degree', 'c_charge_desc', 'is_recid', 'r_case_number', 'r_charge_degree',
                        'r_days_from_arrest', 'r_offense_date', 'r_charge_desc', 'r_jail_in', 'r_jail_out',
                        'violent_recid', 'is_violent_recid', 'vr_case_number', 'vr_charge_degree',
                        'vr_offense_date', 'vr_charge_desc', 'type_of_assessment', 'decile_score.1',
                        'score_text', 'screening_date', 'v_type_of_assessment', 'v_decile_score',
                        'v_score_text', 'v_screening_date', 'in_custody', 'out_custody', 'priors_count.1',#48
                        'start', 'end', 'event', 'two_year_recid']
        """
        """
        features_to_use = ['sex', 'age', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest',
                        'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'is_violent_recid', 'decile_score.1',
                        'v_decile_score', '.1']
        """
        #features_to_use = [5, 7, 9, 10, 11, 12, 13, 14, 15, 21, 22, 23, 24, 32, 33, 39, 43, 48]
        #categorical_features = [5, 9, 10, 11, 12, 13, 14, 22, 23, 24, 32]
        categorical_features = [0, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13]
        transformations = {
            0: lambda x: map_array_values(x, sex_map),
            2: lambda x: map_array_values(x, race_map),
            10: lambda x: map_array_values(x, charge_degree_map),
            11: lambda x: map_array_values(x, charge_desc_map),
        }
        sex_map = {0: 'Female', 1: 'Male'}
        race_map = {0: 'Other', 1: "African-American", 2: 'Caucasian', 3: 'Hispanic', 4: 'Native American', 5: 'Asian'}
        charge_degree_map = {0: 'F', 1: 'M'}
        charge_desc_map = {0: 'Aggravated Assault w/Firearm', 1: 'Felony Battery w/Prior Convict', 2: 'Possession of Cocaine',
                            3: 'Possession of Cannabis', 4: 'arrest case no charge', 5: 'Battery', 6: 'Possession Burglary Tools',
                            7: 'Insurance Fraud', 8: 'Poss 3,4 MDMA (Ecstasy)', 9: 'Poss3,4 Methylenedioxymethcath', 
                            10: 'Felony Driving While Lic Suspd', 11: 'Grand Theft in the 3rd Degree', 12: 'Driving While License Revoked',
                            13: 'Possession Of Heroin', 14: 'Battery on Law Enforc Officer', 15: 'Possession Of Methamphetamine',
                            16: 'Introduce Contraband Into Jail', 17: 'Lewd/Lasc Battery Pers 12+/<16', 18: 'Susp Drivers Lic 1st Offense',
                            19: 'Carrying Concealed Firearm', 20: 'Pos Cannabis W/Intent Sel/Del', 21: 'Tampering With Physical Evidence',
                            22: 'Att Tamper w/Physical Evidence', 23: 'Agg Fleeing and Eluding', 24: 'Operating W/O Valid License',
                            25: 'Poss Wep Conv Felon', 26: 'Possess Cannabis/20 Grams Or Less', 27: 'Unlaw Use False Name/Identity',
                            28: 'Viol Injunct Domestic Violence', 29: 'Defrauding Innkeeper $300/More', 30: 'Uttering a Forged Instrument',
                            31: 'DUI Level 0.15 Or Minor In Veh', 32: 'Driving License Suspended', 33: 'Possession of Oxycodone',
                            34: 'Attempt Armed Burglary Dwell', 35: 'Poss Tetrahydrocannabinols', 36: 'Possess Drug Paraphernalia',
                            37: 'Poss Firearm W/Altered ID#', 38: 'Sell Conterfeit Cont Substance', 39: 'Unlaw LicTag/Sticker Attach',
                            40: 'Aggravated Battery / Pregnant', 41: 'Burglary Structure Unoccup', 42: 'False Name By Person Arrest',
                            43: 'Poss Cocaine/Intent To Del/Sel', 44: 'Burglary Dwelling Assault/Batt', 45: 'Felony Battery (Dom Strang)',
                            46: 'Attempted Burg/struct/unocc', 47: 'Deliver Cocaine', 48: 'Possession Of Alprazolam', 49: 'Flee/Elude LEO-Agg Flee Unsafe',
                            50: 'Fail To Redeliv Hire/Leas Prop', 51: 'Aggravated Assault W/Dead Weap', 52: 'False Ownership Info/Pawn Item',
                            53: 'Possession of Morphine', 54: 'Poss Contr Subst W/o Prescript', 55: 'Aggrav Stalking After Injunctn',
                            56: 'Crim Use of Personal ID Info', 57: 'Resist/Obstruct W/O Violence', 58: 'Petit Theft', 59: 'Disorderly Intoxication', 
                            60: 'Lewdness Violation', 61: 'Poss Pyrrolidinovalerophenone', 62: 'Assault', 63: 'Fail To Obey Police Officer',
                            64: 'Solicit Purchase Cocaine', 65: 'Grand Theft in the 1st Degree', 66: 'Driving Under The Influence',
                            67: 'nan', 68: 'Possession Of Carisoprodol', 69: 'Burglary Conveyance Assault/Bat', 70: 'Deliver 3,4 Methylenediox',
                            71: 'Aggravated Assault W/dead Weap', 72: 'Leave Acc/Attend Veh/More $50', 73: 'Burglary Unoccupied Dwelling', 
                            74: 'Child Abuse', 75: 'Agg Battery Grt/Bod/Harm', 76: 'Lewd or Lascivious Molestation', 77: 'Felony Petit Theft',
                            78: 'Sexual Performance by a Child', 79: 'Leaving Acc/Unattended Veh', 80: 'Fleeing Or Attmp Eluding A Leo', 
                            81: 'Criminal Mischief', 82: 'Aggrav Battery w/Deadly Weapon', 83: 'Trespass Struct/Conveyance', 
                            84: 'DUI Property Damage/Injury', 85: 'Aggravated Battery (Firearm/Actual Possession)', 86: 'Robbery / No Weapon',
                            87: 'Grand Theft (Motor Vehicle)', 88: 'Robbery / Weapon', 89: 'Burglary With Assault/battery', 90: 'Voyeurism',
                            91: 'False Imprisonment', 92: 'Prowling/Loitering', 93: 'Viol Prot Injunc Repeat Viol', 94: 'Throw In Occupied Dwell',
                            95: 'Burglary Conveyance Unoccup', 96: 'Unauth Poss ID Card or DL', 97: 'Opert With Susp DL 2nd Offens', 
                            98: 'Failure To Return Hired Vehicle', 99: 'Agg Fleeing/Eluding High Speed', 100: 'Attempted Robbery  No Weapon',
                            101: 'Resist Officer w/Violence', 102: 'Battery On Parking Enfor Speci', 103: 'Corrupt Public Servant',
                            104: 'Robbery Sudd Snatch No Weapon', 105: 'Forging Bank Bills/Promis Note', 106: 'Felony/Driving Under Influence',
                            107: 'Tamper With Witness/Victim/CI', 108: 'Throw Deadly Missile Into Veh', 109: 'Exposes Culpable Negligence',
                            110: 'Use Scanning Device to Defraud', 111: 'Leaving the Scene of Accident', 112: 'Crimin Mischief Damage $1000+',
                            113: 'Fleeing or Eluding a LEO', 114: 'Possession of Ethylone', 115: 'Aggravated Battery', 116: 'Felony DUI (level 3)',
                            117: 'Fraudulent Use of Credit Card', 118: 'Drivg While Lic Suspd/Revk/Can', 119: 'Burglary Dwelling Occupied',
                            120: 'Cash Item w/Intent to Defraud', 121: 'False Bomb Report', 122: 'Leave Accd/Attend Veh/Less $50', 
                            123: 'Fail Register Vehicle', 124: 'Trespassing/Construction Site', 125: 'Reckless Driving', 
                            126: 'Consp Traff Oxycodone 28g><30k', 127: 'Unemployment Compensatn Fraud', 128: 'Sexual Battery / Vict 12 Yrs +',
                            129: 'Neglect Child / No Bodily Harm', 130: 'Criminal Mischief Damage <$200', 131: 'Aggravated Assault',
                            132: 'Disorderly Conduct', 133: 'Viol Pretrial Release Dom Viol', 134: 'Petit Theft $100- $300', 
                            135: 'Att Burgl Unoccupied Dwel', 136: 'Grand Theft Firearm', 137: 'Failure To Pay Taxi Cab Charge',
                            138: 'Burglary Conveyance Occupied', 139: 'Manslaughter W/Weapon/Firearm', 140: 'Arson II (Vehicle)',
                            141: 'Violation of Injunction Order/Stalking/Cyberstalking', 142: 'Obstruct Fire Equipment', 
                            143: 'Deliver Alprazolam', 144: 'Manufacture Cannabis', 145: 'Attempted Robbery Firearm', 146: 'Fail To Secure Load',
                            147: 'Battery on a Person Over 65', 148: 'Felony Battery', 149: 'Fel Drive License Perm Revoke',
                            150: 'Deliver Cannabis', 151: 'Deliver Cocaine 1000FT Church', 152: 'Possession of Hydromorphone', 
                            153: 'Simulation of Legal Process', 154: 'Defrauding Innkeeper', 155: 'Grand Theft of a Fire Extinquisher',
                            156: 'Fighting/Baiting Animals', 157: 'Att Burgl Conv Occp', 158: 'Depriv LEO of Protect/Communic',
                            159: 'Delivery of 5-Fluoro PB-22', 160: 'Open Carrying Of Weapon', 161: 'Pos Cannabis For Consideration',
                            162: 'Uttering Forged Bills', 163: 'Expired DL More Than 6 Months', 164: 'Stalking', 165: 'Trespass Structure/Conveyance',
                            166: 'DUI - Enhanced', 167: 'Sex Offender Fail Comply W/Law', 168: 'Battery Emergency Care Provide',
                            169: 'Sale/Del Counterfeit Cont Subs', 170: 'Possession Child Pornography', 171: 'Lve/Scen/Acc/Veh/Prop/Damage',
                            172: 'Sex Battery Deft 18+/Vict 11-', 173: 'Posses/Disply Susp/Revk/Frd DL', 174: 'DUI Blood Alcohol Above 0.20',
                            175: 'Burglary Conveyance Armed', 176: 'Crim Attempt/Solicit/Consp', 177: 'License Suspended Revoked',
                            178: 'Live on Earnings of Prostitute', 179: 'Robbery W/Firearm', 180: 'Money Launder 100K or More Dols',
                            181: 'Aggravated Assault W/o Firearm', 182: 'Poss Unlaw Issue Driver Licenc', 183: 'Theft/To Deprive',
                            184: 'Retail Theft $300 1st Offense', 185: 'Intoxicated/Safety Of Another', 186: 'Gambling/Gamb Paraphernalia',
                            187: 'Neglect/Abuse Elderly Person', 188: 'Traffick Amphetamine 28g><200g', 189: 'Grand Theft In The 3Rd Degree',
                            190: 'Poss Of Controlled Substance', 191: 'Del of JWH-250 2-Methox 1-Pentyl', 192: 'Purchasing Of Alprazolam',
                            193: 'Unauthorized Interf w/Railroad', 194: 'Possession Of Lorazepam', 195: 'Restraining Order Dating Viol',
                            196: 'Solic to Commit Battery', 197: 'Carjacking with a Firearm', 198: 'Culpable Negligence', 
                            199: 'Criminal Mischief>$200<$1000', 200: 'Delivery of Heroin', 201: 'DUI - Property Damage/Personal Injury',
                            202: 'Exploit Elderly Person 20-100K', 203: 'Poss of Methylethcathinone', 203: 'Possession Of Buprenorphine',
                            204: 'Tresspass Struct/Conveyance', 205: 'Poss Alprazolam W/int Sell/Del', 206: 'Offer Agree Secure For Lewd Act',
                            207: 'Prostitution/Lewdness/Assign', 208: 'Neglect Child / Bodily Harm', 209: 'Trespass Structure w/Dang Weap',
                            210: 'Possession of Benzylpiperazine', 211: 'Cruelty Toward Child', 212: 'Prostitution/Lewd Act Assignation',
                            213: 'Grand Theft Dwell Property', 214: 'Sound Articles Over 100', 215: 'Burgl Dwel/Struct/Convey Armed',
                            216: 'Ride Tri-Rail Without Paying', 217: 'Disrupting School Function', 218: 'Strong Armed  Robbery',
                            219: 'Poss Trifluoromethylphenylpipe', 220: 'Felony Batt(Great Bodily Harm)', 221: 'Carry Open/Uncov Bev In Pub',
                            222: 'Possession of Hydrocodone', 223: 'Agg Assault Law Enforc Officer', 224: 'Agg Assault W/int Com Fel Dome',
                            225: 'Poss Cntrft Contr Sub w/Intent', 226: 'Counterfeit Lic Plates/Sticker', 227: 'Possession of Butylone',
                            228: 'Sale/Del Cannabis At/Near Scho', 229: 'Poss of Firearm by Convic Felo', 230: 'Refuse to Supply DNA Sample',
                            231: 'Stalking (Aggravated)', 232: 'Sel/Pur/Mfr/Del Control Substa', 233: 'Poss Drugs W/O A Prescription',
                            234: 'Poss of Cocaine W/I/D/S 1000FT Park', 235: 'Lewd Act Presence Child 16-', 236: 'Soliciting For Prostitution',
                            237: 'Extradition/Defendants', 238: 'Refuse Submit Blood/Breath Test', 239: 'Trespass Other Struct/Conve',
                            240: 'Solicit Deliver Cocaine', 241: 'Felon in Pos of Firearm or Amm', 
                            242: 'Burglary Dwelling Armed', 243: 'DWI w/Inj Susp Lic / Habit Off', 244: 'DUI- Enhanced',
                            245: 'Violation License Restrictions', 246: 'Aggr Child Abuse-Torture,Punish', 247: 'Purchase Cannabis',
                            248: 'Use of Anti-Shoplifting Device', 249: 'Possess Countrfeit Credit Card', 250: 'Robbery W/Deadly Weapon',
                            251: 'Poss Counterfeit Payment Inst', 252: 'D.U.I. Serious Bodily Injury', 253: 'Poss Anti-Shoplifting Device',
                            254: 'Threat Public Servant', 255: 'Use Of 2 Way Device To Fac Fel', 256: 'Escape', 257: 'DUI/Property Damage/Persnl Inj',
                            258: 'Hiring with Intent to Defraud', 259: 'Carrying A Concealed Weapon', 260: 'Solicitation On Felony 3 Deg', 
                            261: 'Video Voyeur-<24Y on Child >16', 262: 'Sell or Offer for Sale Counterfeit Goods', 263: 'Throw Missile Into Pub/Priv Dw',
                            264: 'Crim Use Of Personal Id Info', 265: 'Possession Of Diazepam', 266: 'Burglary Structure Assault/Batt', 
                            267: 'Shoot In Occupied Dwell', 268: 'Battery On A Person Over 65', 269: 'Fail To Redeliver Hire Prop', 
                            269: 'Unl/Disturb Education/Instui', 270: 'Violation Of Boater Safety Id', 271: 'False Motor Veh Insurance Card', 
                            272: 'DWLS Susp/Cancel Revoked', 273: 'Viol Injunction Protect Dom Vi', 274: 'Aggrav Child Abuse-Agg Battery',
                            275: 'Deliver Cocaine 1000FT Store', 276: 'Aggravated Battery On 65/Older', 277: 'Possess/Use Weapon 1 Deg Felon',
                            278: 'Fail Obey Driv Lic Restrictions', 279: 'Carjacking w/o Deadly Weapon', 280: 'Contribute Delinquency Of A Minor',
                            281: 'Aggrav Child Abuse-Causes Harm', 282: 'Imperson Public Officer or Emplyee', 283: 'Possession of Codeine',
                            284: 'Tamper With Victim', 285: 'Abuse Without Great Harm', 286: 'Compulsory Sch Attnd Violation', 287: 'Battery On Fire Fighter',
                            288: 'Oper Motorcycle W/O Valid DL', 289: 'Aiding Escape', 290: 'Traffick Hydrocodone   4g><14g', 
                            291: 'Poss/Sell/Del Cocaine 1000FT Sch', 292: 'Poss/pur/sell/deliver Cocaine', 293: 'Del Morphine at/near Park',
                            294: 'Giving False Crime Report', 295: 'Felony Committing Prostitution', 296: 'Possess Tobacco Product Under 18',
                            297: 'Murder in the First Degree', 298: 'Use Computer for Child Exploit', 299: 'Traff In Cocaine <400g>150 Kil',
                            300: 'Murder In 2nd Degree W/firearm', 301: 'Grand Theft (motor Vehicle)', 302: 'Poss Meth/Diox/Meth/Amp (MDMA)',
                            303: 'Trans/Harm/Material to a Minor', 304: 'Harass Witness/Victm/Informnt', 305: 'Grand Theft of the 2nd Degree',
                            306: 'Possession Of Phentermine', 307: 'Poss Of RX Without RX', 308: 'Interference with Custody', 
                            309: 'Traffic Counterfeit Cred Cards', 310: 'Possession Of 3,4Methylenediox', 311: 'Crlty Twrd Child Urge Oth Act',
                            312: 'Dealing in Stolen Property', 313: 'Obtain Control Substance By Fraud', 314: 'Tampering with a Victim',
                            315: 'Poss Pyrrolidinovalerophenone W/I/D/S', 316: 'Solicit To Deliver Cocaine', 317: 'Pos Methylenedioxymethcath W/I/D/S',
                            318: 'Offn Against Intellectual Prop', 319: 'Poss Of 1,4-Butanediol', 320: 'Poss F/Arm Delinq', 321: 'Poss/Sell/Deliver Clonazepam',
                            322: 'Attempted Robbery  Weapon', 323: 'Traffick Oxycodone     4g><14g', 324: 'Interfere W/Traf Cont Dev RR', 
                            325: 'Tresspass in Structure or Conveyance', 326: 'Attempted Burg/Convey/Unocc', 327: 'Att Burgl Struc/Conv Dwel/Occp',
                            328: 'Murder in 2nd Degree', 328: 'Fabricating Physical Evidence', 329: 'DOC/Cause Public Danger', 330: 'Fail Sex Offend Report Bylaw',
                            331: 'Contradict Statement', 332: 'Unlaw Lic Use/Disply Of Others', 333: 'Del 3,4 Methylenedioxymethcath', 334: 'Possession Of Amphetamine',
                            335: 'Discharge Firearm From Vehicle', 336: 'Lease For Purpose Trafficking', 337: 'Lewd/Lasciv Molest Elder Persn', 
                            338: 'Opert With Susp DL 2ND Offense', 339: 'Del Cannabis At/Near Park', 340: 'Burglary Assault/Battery Armed', 
                            341: 'DWLS Canceled Disqul 1st Off', 342: 'Bribery Athletic Contests', 343: 'Grand Theft on 65 Yr or Older',
                            344: 'Crim Attempt/Solic/Consp', 345: 'Poss/Sell/Del/Man Amobarbital', 346: 'Kidnapping / Domestic Violence',
                            347: 'Cruelty to Animals', 348: 'Trespass Private Property', 349: 'Unauth C/P/S Sounds>1000/Audio', 350: 'Obstruct Officer W/Violence',
                            351: 'Cause Anoth Phone Ring Repeat', 352: 'Poss Unlaw Issue Id', 353: 'PL/Unlaw Use Credit Card', 
                            354: 'Possession of LSD', 355: 'Tamper With Witness', 356: 'Possession Of Cocaine', 357: 'Harm Public Servant Or Family',
                            358: 'Possess Cannabis 1000FTSch', 359: 'Consp Traff Oxycodone  4g><14g', 360: 'Consume Alcoholic Bev Pub', 361: 'Shoot Into Vehicle',
                            362: 'Battery Spouse Or Girlfriend', 363: 'Delivery Of Drug Paraphernalia', 364: 'Theft', 365: 'Misuse Of 911 Or E911 System',
                            366: 'Uttering Forged Credit Card', 367: 'Retail Theft $300 2nd Offense', 368: 'Agg Abuse Elderlly/Disabled Adult',
                            369: 'Accessory After the Fact', 370: 'Prostitution', 371: 'Poss Similitude of Drivers Lic', 372: 'Present Proof of Invalid Insur',
                            373: 'Structuring Transactions', 374: 'Principal In The First Degree', 375: 'Assault Law Enforcement Officer', 376: 'Possession Of Fentanyl',
                            377: 'Del Cannabis For Consideration', 378: 'Possess w/I/Utter Forged Bills', 379: 'False Info LEO During Invest',
                            380: 'Possess Mot Veh W/Alt Vin #', 381: 'Possession Of Paraphernalia', 382: 'Criminal Attempt 3rd Deg Felon', 
                            383: 'Possess Weapon On School Prop', 384: 'Possession of Alcohol Under 21', 385: 'Unlicensed Telemarketing', 386: 'Issuing a Worthless Draft',
                            387: 'Conspiracy to Deliver Cocaine', 388: 'Fraud Obtain Food or Lodging', 389: 'Aide/Abet Prostitution Lewdness',
                            390: 'Arson in the First Degree', 391: 'Possession Firearm School Prop', 392: 'Falsely Impersonating Officer',
                            393: 'Poss Oxycodone W/Int/Sell/Del', 394: 'Poss of Vessel w/Altered ID NO', 395:'Poss Pyrrolidinobutiophenone',
                            396: 'Conspiracy Dealing Stolen Prop', 397: 'Felony DUI - Enhanced', 398: 'Aggravated Battery (Firearm)',
                            399: 'False 911 Call', 400: 'Computer Pornography', 401: 'Trespass Property w/Dang Weap', 402: 'Aggress/Panhandle/Beg/Solict',
                            403: 'Sell/Man/Del Pos/w/int Heroin', 404: 'Purchase/P/W/Int Cannabis', 405: 'Uttering Worthless Check +$150',
                            406: 'Deliver Cannabis 1000FTSch', 407: 'Unlawful Conveyance of Fuel',
                            408: 'Fail Register Career Offender', 409: 'Lewd/Lasc Exhib Presence <16yr', 410: 'Armed Trafficking in Cannabis',
                            411: 'Dealing In Stolen Property', 412: 'Trespass On School Grounds', 413: 'Offer Agree Secure/Lewd Act',
                            414: 'Sex Batt Faml/Cust Vict 12-17Y', 415: 'Possession of Methadone', 416: 'Possession Of Clonazepam',
                            417: 'Trespass Struct/Convey Occupy', 418: 'Sell Cannabis', 419: 'Compulsory Attendance Violation',
                            420: 'Possess Controlled Substance', 421: 'Unlawful Use Of Police Badges', 422: 'Manage Busn W/O City Occup Lic',
                            423: 'Deliver Cocaine 1000FT School', 424: 'Sel Etc/Pos/w/Int Contrft Schd', 425: 'Possession Of Anabolic Steroid',
                            426: 'Exhibition Weapon School Prop', 427: 'Purchase Of Cocaine', 428: 'Deliver Cocaine 1000FT Park',
                            429: 'Burglary Structure Occupied', 430: 'Alcoholic Beverage Violation-FL', 431: 'Attempted Deliv Control Subst',
                            432: 'Possession of XLR11', 434: 'Attempt Burglary (Struct)', 435: 'Littering'}
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'compas/compas.csv'), -1, ',',
            categorical_features=categorical_features, 
            discretize=discretize,
            balance=balance, feature_transformations=transformations)
        dataset.class_names = ['Recidiv', 'Disappear']
        dataset.transformations = transformations

    elif dataset_name == 'mortality':
        x_data = pd.read_csv('./dataset/mortality/mortality.csv')
        y_data = x_data['label']
        x_data = x_data.drop(['label'], axis=1)
        feature_names = x_data.columns
        x_data = x_data.to_numpy()
        y_data = y_data.to_numpy()
        categorical_features = [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 17, 18,
            24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42]        
        continuous_features = [2, 3, 13, 14, 15, 19, 20, 21, 22, 23, 36, 43, 44, 45]

        categorical_names = {}
        for feature in categorical_features:
            le = sklearn.preprocessing.LabelEncoder()
            le.fit(x_data[:, feature])
            x_data[:, feature] = le.transform(x_data[:, feature])
            categorical_names[feature] = [str(x) for x in le.classes_]
            
        data = x_data.astype(float)
        ordinal_features = []
        if discretize:
            disc = limes.lime_tabular.QuartileDiscretizer(data,
                                                        categorical_features,
                                                        feature_names)
            data = disc.discretize(data)
            ordinal_features = [x for x in range(data.shape[1])
                                if x not in categorical_features]
            categorical_features = range(data.shape[1])
            categorical_names.update(disc.names)
        for x in categorical_names:
            categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
        
        categorical_values =[]
        for nb, features in enumerate(categorical_features):
            try:
                tab = list(set(x_data[:,features]))
            except ValueError:
                tab = [i for i in range(len(categorical_names[features]))]
            if not 0 in tab:
                tab.insert(0, 0)
            categorical_values.append(tab)
        
        dataset = Bunch({})
        dataset.train, dataset.labels_train = x_data, y_data
        dataset.categorical_features, dataset.continuous_features = categorical_features, continuous_features
        dataset.class_names = ['surviving', 'not surviving']
        dataset.feature_names = feature_names
        dataset.categorical_names, dataset.categorical_values = categorical_names, categorical_values
        
    elif dataset_name == 'blood':
        feature_names = ["Recency", "Frequency",  "Monetary", "Time", "Class"]
        
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'blood_transfusion_service.csv'), -1, ', ',
            feature_names=feature_names, 
            discretize=discretize, skip_first=True,
            balance=balance)
        dataset.class_names = ['Donated', 'Not donated']

    elif dataset_name == 'diabetes':
        feature_names = ["Pregnancies", "Glucose",  "Blood pressure", "Skin Thickness", "Insulin",
                        "BMI", "Diabetes Pedigree Function", "Age", "Outcome"]
        
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'diabetes.csv'), -1, ', ',
            feature_names=feature_names, 
            discretize=discretize, skip_first=True,
            balance=balance)
        dataset.class_names = ['Tested positive', 'Tested negative']

    elif 'generate' in dataset_name:
        feature_names = ["x", "y", "class"]
        
        dataset = load_csv_dataset(
            os.path.join(dataset_folder, 'generate/generate.data'), -1, ', ',
            feature_names=feature_names, discretize=discretize, balance=balance)
        dataset.class_names = ['class 0', 'class 1']
    return dataset



def load_csv_dataset(data, target_idx, delimiter=',',
                     feature_names=None, categorical_features=None,
                     features_to_use=None, feature_transformations=None,
                     discretize=False, balance=False, fill_na='-1', filter_fn=None, skip_first=False, data_generate=None):
    """if not feature names, takes 1st line as feature names
    if not features_to_use, use all except for target
    if not categorical_features, consider everything < 20 as categorical"""
    if feature_transformations is None:
        feature_transformations = {}
    if data_generate == None:
        if "blood" in data:
            data = np.genfromtxt("dataset/blood_transfusion_service.csv", delimiter=",", dtype='|S128')
        elif "diabetes" in data:
            data = np.genfromtxt("dataset/diabetes.csv", delimiter=",", dtype='|S128')
        #elif "compas" in data:

        else:
            try:
                data = np.genfromtxt(data, delimiter=delimiter, dtype='|S128')
            except:
                import pandas
                data = pandas.read_csv(data,
                               header=None,
                               delimiter=delimiter,
                               na_filter=True,
                               dtype=str).fillna(fill_na).values
    if target_idx < 0:
        target_idx = data.shape[1] + target_idx
    ret = Bunch({})
    if feature_names is None:
        feature_names = list(data[0])
        data = data[1:]
    else:
        feature_names = copy.deepcopy(feature_names)
    if skip_first:
        data = data[1:]
    if filter_fn is not None:
        data = filter_fn(data)
    
    for feature, fun in feature_transformations.items():
        data[:, feature] = fun(data[:, feature])
    labels = data[:, target_idx]
    le = sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    ret.labels = le.transform(labels)
    labels = ret.labels
    ret.class_names = list(le.classes_)
    ret.class_target = feature_names[target_idx]
    if features_to_use is not None:
        data = data[:, features_to_use]
        feature_names = ([x for i, x in enumerate(feature_names)
                          if i in features_to_use])
        if categorical_features is not None:
            categorical_features = ([features_to_use.index(x)
                                     for x in categorical_features])
    else:
        data = np.delete(data, target_idx, 1)
        feature_names.pop(target_idx)
        if categorical_features:
            categorical_features = ([x if x < target_idx else x - 1
                                     for x in categorical_features])

    if categorical_features is None:
        categorical_features = []
        for f in range(data.shape[1]):
            if len(np.unique(data[:, f])) < 20:
                categorical_features.append(f)
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_
    data = data.astype(float)
    ordinal_features = []
    if discretize:
        disc = lime.lime_tabular.QuartileDiscretizer(data,
                                                     categorical_features,
                                                     feature_names)
        data = disc.discretize(data)
        ordinal_features = [x for x in range(data.shape[1])
                            if x not in categorical_features]
        categorical_features = range(data.shape[1])
        categorical_names.update(disc.names)
    for x in categorical_names:
        categorical_names[x] = [y.decode() if type(y) == np.bytes_ else y for y in categorical_names[x]]
    ret.ordinal_features = ordinal_features
    ret.categorical_features = categorical_features
    ret.categorical_names = categorical_names
    ret.feature_names = feature_names
    np.random.seed(1)
    if balance:
        idxs = np.array([], dtype='int')
        min_labels = np.min(np.bincount(labels))
        for label in np.unique(labels):
            idx = np.random.choice(np.where(labels == label)[0], min_labels)
            idxs = np.hstack((idxs, idx))
        data = data[idxs]
        labels = labels[idxs]
        ret.data = data
        ret.labels = labels

    splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                  test_size=.2,
                                                  random_state=1)
    train_idx, test_idx = [x for x in splits.split(data)][0]
    ret.train = data[train_idx]
    ret.labels_train = ret.labels[train_idx]
    cv_splits = sklearn.model_selection.ShuffleSplit(n_splits=1,
                                                     test_size=.5,
                                                     random_state=1)
    cv_idx, ntest_idx = [x for x in cv_splits.split(test_idx)][0]
    cv_idx = test_idx[cv_idx]
    test_idx = test_idx[ntest_idx]
    ret.validation = data[cv_idx]
    ret.labels_validation = ret.labels[cv_idx]
    ret.test = data[test_idx]
    ret.labels_test = ret.labels[test_idx]
    ret.test_idx = test_idx
    ret.validation_idx = cv_idx
    ret.train_idx = train_idx

    # ret.train, ret.test, ret.labels_train, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(data, ret.labels,
    #                                               train_size=0.80))
    # ret.validation, ret.test, ret.labels_validation, ret.labels_test = (
    #     sklearn.cross_validation.train_test_split(ret.test, ret.labels_test,
    #                                               train_size=.5))
    
    # Code to save Compas modifed data into a csv file 
    """ 
    np.set_printoptions(precision=3)
    data = data.astype('str')
    data = np.insert(data, 0, ['sex', 'age', 'race', 'juv_fel_count', 'decile_score', 'juv_misd_count', 'juv_other_count', 'priors_count', 'days_b_screening_arrest',
                        'c_days_from_compas', 'c_charge_degree', 'c_charge_desc', 'is_recid', 'is_violent_recid', 'decile_score.1',
                        'v_decile_score', 'priors_count.1', 'two_year_recid'], 0)
    print("data numy", data)
    np.savetxt("dataset/compas/compas_numpy.csv", data, delimiter=",", fmt='%s')
    """

    ret.data = data
    return ret

class Neighbors:
    def __init__(self, nlp_obj):
        self.nlp = nlp_obj
        self.to_check = [w for w in self.nlp.vocab if w.prob >= -15]
        self.n = {}

    def neighbors(self, word):
        word = unicode(word)
        orig_word = word
        if word not in self.n:
            if word not in self.nlp.vocab:
                self.n[word] = []
            else:
                word = self.nlp.vocab[unicode(word)]
                queries = [w for w in self.to_check
                            if w.is_lower == word.is_lower]
                if word.prob < -15:
                    queries += [word]
                by_similarity = sorted(
                    queries, key=lambda w: word.similarity(w), reverse=True)
                self.n[orig_word] = [(self.nlp(w.orth_)[0], word.similarity(w))
                                     for w in by_similarity[:500]]
                                    #  if w.lower_ != word.lower_]
        return self.n[orig_word]

def perturb_sentence(text, present, n, neighbors, proba_change=0.5,
                     top_n=50, forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True,
                     temperature=.4):
    # words is a list of words (must be unicode)
    # present is which ones must be present, also a list
    # n = how many to sample
    # neighbors must be of utils.Neighbors
    # nlp must be spacy
    # proba_change is the probability of each word being different than before
    # forbidden: forbidden lemmas
    # forbidden_tags, words: self explanatory
    # pos: which POS to change

    tokens = neighbors.nlp(unicode(text))
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    raw = np.zeros((n, len(tokens)), '|S80')
    data = np.ones((n, len(tokens)))
    raw[:] = [x.text for x in tokens] # This line replace all element in the array raw to get
                                      # the value of the sentence
    for i, t in enumerate(tokens):
        if i in present:
            continue
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            # Returns words that have the same tag (i.e: Nouns, adj, etc...) 
            # among the 500 words that are most similar to the word in entry
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                # print sorted(zip(t_neighbors, weights), key=lambda x:x[1], reverse=True)[:10]
                raw[:, i] = np.random.choice(t_neighbors, n,  p=weights,
                                             replace=True)
                # The type of data in raw is byte.
                data[:, i] = raw[:, i] == t.text.encode()
            else:
                n_changed = np.random.binomial(n, proba_change)
                changed = np.random.choice(n, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
#         else:
#             print t.text, t.pos_ in pos, t.lemma_ in forbidden, t.tag_ in forbidden_tags, t.text in neighbors
    if (sys.version_info > (3, 0)):
        raw = [' '.join([y.decode() for y in x]) for x in raw]
    else:
        raw = [' '.join(x) for x in raw]
    return raw, data

def return_pertinent_sentences(pertinent, raw_data, m):
    """
    Generates all the sentences generated during perturbation with add of the pertinent negative words
    """
    pertinent_sentences = np.zeros((m, len(raw_data)), '|S80')
    for i, t in enumerate(raw_data):
        for j in range(m):
            if pertinent[j][i] == 1:
                pertinent_sentences[j][i] = raw_data[i]
            else:
                pertinent_sentences[j][i] = ""
    if (sys.version_info > (3, 0)):
        raw = []
        for x in pertinent_sentences:
            text = " "
            for y in x:
                if y.decode():
                    text+= " " + ' '.join([y.decode()])
            raw.append(text)
    else:
        raw = [' '.join(x) for x in pertinent_sentences]
    return raw

def generate_false_pertinent(text, present, m, neighbors, n_best_co_occurrence, proba_change=0.5,
                     forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'],
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True, generate_sentence=False):
    """ 
    Generates a matrix composed of sentence with the 'false pertinent' that represents words that frequently co occur
    args:
        present is which ones must be present, also a list
        m = how many to sample
        neighbors must be of utils.Neighbors
        n_best_co_occurrence: The matrix of the n words that most frequently co occurs
        nlp must be spacy
        proba_change is the probability of each word being different than before
        forbidden: forbidden lemmas
        forbidden_tags, words: self explanatory
        words is a list of words (must be unicode)
        pos: which POS to change
        generate_sentence: If set to True, return the sentence composed of all the pertinent negatifs words 
    """
    # Use of classical natural language processing
    tokens = neighbors.nlp(unicode(text))
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    sentence = []
    for x in tokens:
        sentence.append(x.text)  
    pertinent = np.zeros(m)
    array_false_pertinent = []
    for i, t in enumerate(sentence):
        array_false_pertinent.append(t.encode('ascii'))
        # gets the most frequent words associated with the target word t
        targets = co_occ.generate_bi_grams_words(t, n_best_co_occurrence)
        # Put to 1 for all sentence generated at the position of the word from the target sentence
        pertinent = np.c_[pertinent, np.ones(m)]
        if targets != []:
            # Add randomly a 1 in the matrix for (only) one of the most co occurent words 
            size_pertinents = len(targets)
            matrix_raw_false_pertinent = np.zeros((m, size_pertinents))
            for j, p in enumerate(targets):
                array_false_pertinent.append(p.encode('ascii'))
            k = 0
            for i in range(m):
                matrix_raw_false_pertinent[i][k] = 1
                k += 1
                k = k % size_pertinents
            np.random.shuffle(matrix_raw_false_pertinent)
            pertinent = np.c_[pertinent, matrix_raw_false_pertinent]
    if generate_sentence:
        # generates a sentence composed of all the pertinent negatifs words inside the target sentence
        sentence_false_pertinent = ""
        for word in array_false_pertinent:
            sentence_false_pertinent += " " + word.decode()
        return sentence_false_pertinent
    pertinent = np.delete(pertinent, 0, 1)  
    raw = return_pertinent_sentences(pertinent, array_false_pertinent, m)
    return pertinent, raw, array_false_pertinent

def return_pertinent_sentences_replace(pertinent, raw_data, m, raw, array_replace_words):
    """
    Generates all the sentences generated during perturbation with add of the pertinent negative words
    """
    pertinent_sentences = np.zeros((m, len(raw_data)), '|S80')
    counter = 0
    for i, t in enumerate(raw_data):
        if i in array_replace_words:
            pertinent_sentences[:, i] = raw[:,counter]
            counter += 1
        else :
            for j in range(m):
                if pertinent[j][i] == 1:
                    pertinent_sentences[j][i] = t
                else:
                    pertinent_sentences[j][i] = ""
    if (sys.version_info > (3, 0)):
        raw = []
        for x in pertinent_sentences:
            text = " "
            for y in x:
                if y.decode():
                    text+= " " + ' '.join([y.decode()])
            raw.append(text)
    else:
        raw = [' '.join(x) for x in pertinent_sentences]
    return raw

def generate_false_pertinents_replace(text, present, m, neighbors, n_best_co_occurrence, proba_change=0.5,
                     forbidden=[], forbidden_tags=['PRP$'],
                     forbidden_words=['be'], top_n=50, temperature=.4,
                     pos=['NOUN', 'VERB', 'ADJ', 'ADV', 'ADP', 'DET'], use_proba=True, generate_sentence=False):
    """ 
    Generates a matrix composed of sentence with the 'false pertinent' that represents words that frequently co occur
    args:
        present is which ones must be present, also a list
        m = how many to sample
        neighbors must be of utils.Neighbors
        n_best_co_occurrence: The matrix of the n words that most frequently co occurs
        nlp must be spacy
        proba_change is the probability of each word being different than before
        forbidden: forbidden lemmas
        forbidden_tags, words: self explanatory
        words is a list of words (must be unicode)
        pos: which POS to change
        generate_sentence: If set to True, return the sentence composed of all the pertinent negatifs words 
    """
    # Use of classical natural language processing
    tokens = neighbors.nlp(unicode(text))
    forbidden = set(forbidden)
    forbidden_tags = set(forbidden_tags)
    forbidden_words = set(forbidden_words)
    pos = set(pos)
    sentence = []
    for x in tokens:
        sentence.append(x.text)  
    pertinent = np.zeros(m)
    array_false_pertinent = []
    raw = np.zeros((m, len(tokens)), '|S80')
    data = np.ones((m, len(tokens)))
    raw[:] = [x.text for x in tokens]
    array_replace_words = []
    counter = 0
    for i, t in enumerate(tokens):
        if (t.text not in forbidden_words and t.pos_ in pos and
                t.lemma_ not in forbidden and t.tag_ not in forbidden_tags):
            # Returns words that have the same tag (i.e: Nouns, adj, etc...) 
            # among the 500 words that are most similar to the word in entry
            r_neighbors = [
                (unicode(x[0].text.encode('utf-8'), errors='ignore'), x[1])
                for x in neighbors.neighbors(t.text)
                if x[0].tag_ == t.tag_][:top_n]
            if not r_neighbors:
                continue
            t_neighbors = [x[0] for x in r_neighbors]
            weights = np.array([x[1] for x in r_neighbors])
            if use_proba:
                weights = weights ** (1. / temperature)
                weights = weights / sum(weights)
                # print sorted(zip(t_neighbors, weights), key=lambda x:x[1], reverse=True)[:10]
                raw[:, i] = np.random.choice(t_neighbors, m,  p=weights,
                                             replace=True)
                # The type of data in raw is byte.
                data[:, i] = raw[:, i] == t.text.encode()
            else:
                n_changed = np.random.binomial(m, proba_change)
                changed = np.random.choice(m, n_changed, replace=False)
                if t.text in t_neighbors:
                    idx = t_neighbors.index(t.text)
                    weights[idx] = 0
                weights = weights / sum(weights)
                raw[changed, i] = np.random.choice(t_neighbors, n_changed, p=weights)
                data[changed, i] = 0
        #t = t.decode('ascii')
        array_false_pertinent.append(t.text.encode('ascii'))
        # gets the most frequent words associated with the target word t
        targets = co_occ.generate_bi_grams_words(t.text, n_best_co_occurrence)
        # Put to 1 for all sentence generated at the position of the word from the target sentence
        # pertinent = np.c_[pertinent, np.ones(m)]
        pertinent = np.c_[pertinent, data[:,i]]
        array_replace_words.append(counter)
        counter += len(targets) + 1
        if targets != []:
            # Add randomly a 1 in the matrix for (only) one of the most co occurent words 
            size_pertinents = len(targets)
            matrix_raw_false_pertinent = np.zeros((m, size_pertinents))
            for j, p in enumerate(targets):
                array_false_pertinent.append(p.encode('ascii'))
            k = 0
            for i in range(m):
                matrix_raw_false_pertinent[i][k] = 1
                k += 1
                k = k % size_pertinents
            np.random.shuffle(matrix_raw_false_pertinent)
            pertinent = np.c_[pertinent, matrix_raw_false_pertinent]
    if generate_sentence:
        # generates a sentence composed of all the pertinent negatifs words inside the target sentence
        sentence_false_pertinent = ""
        for word in array_false_pertinent:
            sentence_false_pertinent += " " + word.decode()
        return sentence_false_pertinent
    pertinent = np.delete(pertinent, 0, 1)  
    raw = return_pertinent_sentences_replace(pertinent, array_false_pertinent, m, raw, array_replace_words)
    return pertinent, raw, array_false_pertinent

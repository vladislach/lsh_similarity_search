import math
import sys
import functools
from rdkit import Chem
from rdkit.Chem import AllChem
from bson import ObjectId
from rdchiral.template_extractor import extract_from_reaction
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun


# Default configurations for a variety of constants.
DEFAULT_THRESHOLD = 0.1

# Morgan fingerprints
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_LEN = 2048

# LSH constants
DEFAULT_BIT_N = 2048
DEFAULT_BUCKET_N = 25

def calc_tanimoto(Na, Nb):
    """Calculates the Tanimoto similarity coefficient between two sets NA and NB."""
    Nab = len(set(Na).intersection((set(Nb))))
    return float(Nab) / (len(Na) + len(Nb) - Nab)

def get_min_hash(mol, permutations):
    qfp_bits = [int(n) for n in list(AllChem.GetMorganFingerprintAsBitVect(mol, DEFAULT_MORGAN_RADIUS, nBits=DEFAULT_MORGAN_LEN))]
    min_hash = []
    for perm in permutations:
        for idx, i in enumerate(perm):
            if qfp_bits[i]:
                min_hash.append(idx)
                break
    return min_hash

def hash_to_buckets(min_hash, num_buckets=DEFAULT_BUCKET_N, nBits=DEFAULT_BIT_N):
    if len(min_hash) % num_buckets:
        raise Exception('number of buckets must be divisiable by the hash length')
    buckets = []
    hash_per_bucket = int(len(min_hash) / num_buckets)
    num_bits = (nBits-1).bit_length()
    if num_bits * hash_per_bucket > sys.maxsize:
        raise Exception('numbers are too large to produce valid buckets')
    for b in range(num_buckets):
        buckets.append(functools.reduce(lambda x, y: (x << num_bits) + y, min_hash[b:(b + hash_per_bucket)]))
    return buckets


def sim_search(mol, mol_collection, count_collection=None, threshold=DEFAULT_THRESHOLD):
    """
    Searches `mol_collection` for molecules with Tanimoto similarity to `mol`
    greater than or equal to `threshold`.
    """
    qfp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits())
    qn = len(qfp)
    qmin = int(math.ceil(qn * threshold))
    try:
        qmax = int(qn / threshold)
    except ZeroDivisionError:
        qmax = float("inf")
    ncommon = qn - qmin + 1

    if count_collection is not None:
        reqbits = [count['_id'] for count in count_collection.find({'_id': {'$in': qfp}}).sort('count', 1).limit(ncommon)]
    else:
        reqbits = qfp[:ncommon]

    results = []
    for mol in mol_collection.find({'mfp.bits': {'$in': reqbits}, 'mfp.count': {'$gte': qmin, '$lte': qmax}}):
        tanimoto = calc_tanimoto(qfp, mol["mfp"]["bits"])
        mol["tanimoto"] = tanimoto
        if tanimoto >= threshold:
                results.append((mol['smiles'], tanimoto))
    return sorted(results, key=lambda x: x[1], reverse=True)


def sim_search_aggregate(mol, mol_collection, count_collection=None, threshold=DEFAULT_THRESHOLD):
    """
    Searches `mol_collection` for molecules with Tanimoto similarity to `mol`
    greater than or equal to `threshold`.
    This method uses a MongoDB aggregation pipeline.
    """
    qfp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits())
    qn = len(qfp)
    qmin = int(math.ceil(qn * threshold))
    try:
        qmax = int(qn / threshold)
    except ZeroDivisionError:
        qmax = float("inf")
    ncommon = qn - qmin + 1

    if count_collection is not None:
        reqbits = [count['_id'] for count in count_collection.find({'_id': {'$in': qfp}}).sort('count', 1).limit(ncommon)]
    else:
        reqbits = qfp[:ncommon]

    aggregate = [
        {'$match': {'mfp.count': {'$gte': qmin, '$lte': qmax}, 'mfp.bits': {'$in': reqbits}}},
        {'$set': {
            'tanimoto': {'$let':
                {'vars': {'common': {'$size': {'$setIntersection': ['$mfp.bits', qfp]}}},
                'in': {'$divide': ['$$common', {'$subtract': [{'$add': [qn, '$mfp.count']}, '$$common']}]}}
                }
            }
        },
        {'$match': {'tanimoto': {'$gte': threshold}}},
        {"$sort": {"tanimoto": -1}}
    ]
    
    response = mol_collection.aggregate(aggregate)
    output = [{'smiles': i['smiles'], 'tanimoto': i['tanimoto'], 'id': i['_id']} for i in response]
    return output


def sim_search_lsh(mol, db, mol_collection, perm_collection, count_collection, threshold=DEFAULT_THRESHOLD):
    qfp = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048).GetOnBits())
    qn = len(qfp)
    qmin = int(math.ceil(qn * threshold))
    try:
        qmax = int(qn / threshold)
    except ZeroDivisionError:
        qmax = float("inf")
    ncommon = qn - qmin + 1

    if count_collection is not None:
        reqbits = [count['_id'] for count in count_collection.find({'_id': {'$in': qfp}}).sort('count', 1).limit(ncommon)]
    else:
        reqbits = qfp[:ncommon]

    permutations = [p['permutation'] for p in perm_collection.find()]
    min_hash = get_min_hash(mol, permutations)
    hash_groups = hash_to_buckets(min_hash)
    nested_res = []
    cursors = [db['lsh_' + str(i)].find({'_id': h}, {'molecules': 1}) for i, h in enumerate(hash_groups)]

    for c in cursors:
        cursor = list(c)
        if len(cursor) == 0:
            continue
        else:
            nested_res.append(cursor[0]['molecules'])

    hashed_ids = []
    for sublist in nested_res:
        for item in sublist:
            try:
                obj_id = ObjectId(item)
                hashed_ids.append(obj_id)
            except:
                pass

    aggregate = [
        {'$match': {'_id': {'$in': hashed_ids},
                    'mfp.count': {'$gte': qmin, '$lte': qmax},
                    'mfp.bits': {'$in': reqbits}}},
        {'$set': {
            'tanimoto': {'$let':
                {'vars': {'common': {'$size': {'$setIntersection': ['$mfp.bits', qfp]}}},
                'in': {'$divide': ['$$common', {'$subtract': [{'$add': [qn, '$mfp.count']}, '$$common']}]}}
                }
            }
        },
        {'$match': {'tanimoto': {'$gte': threshold}}},
        {"$sort": {"tanimoto": -1}}
    ]

    response = mol_collection.aggregate(aggregate)
    output = [{'smiles': i['smiles'], 'tanimoto': i['tanimoto'], 'id': i['_id']} for i in response]
    return output


def find_similar(product_smiles, molecules, counts, reactions, threshold=0.5, num_similar=5):
    mol = Chem.MolFromSmiles(product_smiles)
    similar_products = sim_search_aggregate(mol, molecules, counts, threshold)
    k = max(num_similar, len(similar_products))
    top_reactions_ids = [i['id'] for i in similar_products[:k]]
    top_reactions = [reactions.find_one({'_id': i}) for i in top_reactions_ids]
    top_reactions_templates = [extract_from_reaction(r) for r in top_reactions]

    precursors = []
    for template in top_reactions_templates:
        retro_reaction = rdchiralReaction(template["reaction_smarts"])
        reactants = rdchiralReactants(product_smiles)

        try:
            outcome = rdchiralRun(retro_reaction, reactants, combine_enantiomers=True)
        except Exception as e:
            print(e)
            outcome = []
        
        precursors.append(outcome)

    output = [(i['tanimoto'], j) for i, j in zip(similar_products, precursors)]
    return output


if __name__ == '__main__':
    import pymongo

    # URI = ""
    client = pymongo.MongoClient(URI)
    db = client.similarity_search
    molecules = db.molecules
    counts = db.counts
    reactions = db.reactions
    LSHashes = client.LSHashes
    permutations = LSHashes.permutations

    smiles = "[O:1]=[C:2]([OH:3])[C@H:4]([NH:5][C:8](=[O:7])[C@H:9]([NH2:10])[CH3:11])[CH3:6]"
    mol = Chem.MolFromSmiles(smiles)
    threshold = 0.5

    print("Results from sim_search:")
    similar_molecules = sim_search(mol, molecules, counts, threshold)
    for m in similar_molecules:
        print(m)
    print()

    print("Results from sim_search_aggregate:")
    similar_molecules = sim_search_aggregate(mol, molecules, counts, threshold)
    for m in similar_molecules:
        print(m)
    print()

    print("Results from sim_search_lsh:")
    similar_molecules = sim_search_lsh(mol, LSHashes, molecules, permutations, counts, threshold)
    for m in similar_molecules:
        print(m)
    print()

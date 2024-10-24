from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import sys
import functools


# Morgan fingerprints
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_MORGAN_LEN = 2048

# LSH constants
DEFAULT_BIT_N = 2048
DEFAULT_BUCKET_N = 25
DEFAULT_PERM_LEN = 2048
DEFAULT_PERM_N = 100

# PyMongo configurations
DEFAULT_BATCH_SIZE = 100

def get_permutations(len_permutations=DEFAULT_PERM_LEN, num_permutations=DEFAULT_PERM_N):
    """Gets NUM_PERMUTATIONS random permutations of numbers of length LEN_PERMUTATIONS each."""
    return map(lambda _: np.random.permutation(len_permutations), range(num_permutations))


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


def AddRandomPermutations(perm_collection, len=DEFAULT_PERM_LEN, num=DEFAULT_PERM_N):
    """
    Uses the function get_permutations to generate NUM random permutations
    of bits of length LEN and saves each in COLLECTION as a separate document.
    :param collection: A MongoDB collection.
    :param len: Length of fingerprints to generate permutations for.
    :param num: Number of random permutations to save.
    :return: None
    """
    perm_collection.insert_many([{'_id': i, 'permutation': perm.tolist()} for i, perm in enumerate(get_permutations(len, num))])


def AddLocalityHashes(mol_collection, perm_collection, nBuckets=DEFAULT_BUCKET_N):
    """
    Adds locality-sensitive hash values to each document in MOL_COLLECTION
    based on permutations in PERM_COLLECTION. This method requires documents
    in PERM_COLLECTION to have a 'permutation' field and documents in
    MOL_COLLECTION to have a 'rdmol' field.
    :param mol_collection: A MongoDB collection.
    :param perm_collection: A MongoDB collection.
    :return: None
    """
    length = len(perm_collection.find_one()['permutation'])
    permutations = []
    for doc in perm_collection.find(batch_size=DEFAULT_BATCH_SIZE):
        if len(doc['permutation']) != length:
            Exception('Permutations must split evenly into buckets')
            return None
        permutations.append(doc['permutation'])
    for moldoc in mol_collection.find(batch_size=DEFAULT_BATCH_SIZE):
        mol = Chem.MolFromSmiles(moldoc['smiles'])
        min_hash = get_min_hash(mol, permutations)
        hash_groups = hash_to_buckets(min_hash, nBuckets, length)
        mol_collection.update_one({'_id': moldoc['_id']}, {'$set': {'lsh': hash_groups}})


def AddHashCollections(db, mol_collection):
    """
    Creates different collections in DB that each store a subset of molecules
    in MOL_COLLECTION hashed through LSH.
    :param mol_collection: a MongoDB collection.
    :return: None
    """
    for moldoc in mol_collection.find(batch_size=DEFAULT_BATCH_SIZE):
        hash_groups = moldoc['lsh']
        for number, hash in enumerate(hash_groups):
            db['lsh_' + str(number)].update_one({'_id': hash}, {'$push': {'molecules': moldoc['_id']}}, True)

from pymongo import MongoClient
from rdkit import Chem
from rdkit.Chem import AllChem
from locality_sensitive_hashes import AddRandomPermutations, AddLocalityHashes, AddHashCollections
from simsearch_functions import sim_search, sim_search_aggregate, sim_search_lsh, find_similar


class SimilaritySearch:
    def __init__(self, database=None, molecules_collection=None, reactions_collection=None):
        """
        Initialize a SimilaritySearch instance with a MongoDB database and a molecules collection
        (for molecular similarity search) or a reactions collection (for retrosynthetic similarity search)
        if they are passed as arguments. Otherwise, initialize with None and use connect_and_load to
        initialize a connection to the default MongoDB database and load collections.

        :param database: A MongoDB database.
        :param molecules_collection: A MongoDB collection.
        :param reactions_collection: A MongoDB collection.
        """
        self.db = database
        self.molecules = molecules_collection
        self.reactions = reactions_collection


    def connect_and_load(self):
        """
        Initialize a connection to the MongoDB database and load collections
        if database, molecules_collection, and reactions_collection were not
        passed as arguments to __init__.
        """
        # URI = 
        client = MongoClient(URI)
        self.db = client["test"]
        self.molecules = self.db["molecules"]
        self.reactions = self.db["reactions"]
        self.counts = self.db["counts"]
        self.permutations = self.db["permutations"]


    def get_products_from_reactions(self):
        """
        Extracts the products from each reaction in the reactions collection
        and stores them in the molecules collection to use for similarity search.
        Assumes that the database and the reactions collection have already been
        initialized.
        """
        self.molecules = self.db["molecules"]
        for rxn in self.reactions.find():
            smiles = rxn["products"]
            id = rxn["_id"]
            self.molecules.insert_one({"_id": id, "smiles": smiles})


    def precompute_mols(self):
        """
        Precomputes molecular fingerprints and pattern fingerprints, as well as
        their corresponding bit counts (stored in the counts collection), for each
        molecule in the molecules collection.
        """
        self.counts = self.db["counts"]
        mfp_counts = {}
        for mol in self.molecules.find():
            smiles = mol["smiles"]
            rdmol = Chem.MolFromSmiles(smiles)
            mfp = list(AllChem.GetMorganFingerprintAsBitVect(rdmol, radius=2, nBits=2048).GetOnBits())
            pfp = list(Chem.rdmolops.PatternFingerprint(rdmol).GetOnBits())
            smiles = Chem.MolToSmiles(rdmol)

            for bit in mfp:
                mfp_counts[bit] = mfp_counts.get(bit, 0) + 1
            
            # update the molecule document with the fingerprints and their counts
            self.molecules.update_one({"_id": mol["_id"]}, 
                                      {"$set": {"mfp": {"bits": mfp, "count": len(mfp)},
                                                "pfp": {"bits": pfp, "count": len(pfp)}}})

        for k, v in mfp_counts.items():
            self.counts.insert_one({"_id": k, "count": v})
        
        self.molecules.create_index("mfp.bits")
        self.molecules.create_index("mfp.count")
        self.molecules.create_index("pfp.bits")
        self.molecules.create_index("pfp.count")


    def add_lsh(self):
        """
        Computes and adds locality-sensitive hashes (LSH) to the database
        """
        self.permutations = self.db["permutations"]
        AddRandomPermutations(self.permutations)
        AddLocalityHashes(self.molecules, self.permutations)
        AddHashCollections(self.db, self.molecules)

    def preprocess(self):
        """
        Precomputes fingerprints and LSH for each molecule in the molecules collection.
        """
        self.get_products_from_reactions()
        self.precompute_mols()
        self.add_lsh()


    def find_similar_molecules(self, smiles, similarity_type="mfp", fast=False, threshold=0.5):
        """
        Finds molecules in the molecules collection that are similar to the
        molecule with the given SMILES string.

        :param smiles: A SMILES string.
        :param similarity_type: A string that specifies the type of similarity
                                to use. Can be "mfp" for molecular fingerprints
                                or "pfp" for pattern fingerprints.
        :param fast: A boolean that specifies whether to use locality-sensitive hashes.
        :param threshold: A float that specifies the similarity threshold.
        """
        mol = Chem.MolFromSmiles(smiles)
        if fast:
            return sim_search_lsh(mol, self.db, self.molecules, self.permutations, self.counts, threshold)
        else:
            return sim_search_aggregate(mol, self.molecules, self.counts, threshold)
        

    def find_similar_retrosim(self, smiles, threshold=0.3, top_k=10):
        """
        Finds top_k reactions in the reactions collection that generate products similar to the
        the given SMILES string. Returns the predicted precursors for each reaction and 
        the similarity score.

        :param smiles: A SMILES string.
        :param threshold: A float that specifies the similarity threshold.
        """
        return find_similar(smiles, self.molecules, self.counts, self.reactions, threshold, top_k)


if __name__ == '__main__':
    searcher = SimilaritySearch()
    searcher.connect_and_load()
    searcher.molecules.find_one()
    print(searcher.find_similar_molecules("O=C/C=C/C1=CC=COC1O", fast=False, threshold=0.3))
    print(searcher.find_similar_retrosim("[O:1]=[CH:2]/[CH:3]=[CH:4]/[C:5]1=[CH:8][CH:9]=[C:10]([C:12](=[O:13])[OH:14])[O:11][CH:6]1[OH:7]", threshold=0.3, top_k=10))
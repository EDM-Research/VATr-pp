import os


def test_split():
    iam_path = r"C:\Users\bramv\Documents\Werk\Research\Unimore\datasets\IAM"

    original_set_names = ["trainset.txt", "validationset1.txt", "validationset2.txt", "testset.txt"]
    original_set_ids = []

    print("ORIGINAL IAM")
    print("---------------------")

    for set_name in original_set_names:
        with open(os.path.join(iam_path, set_name), 'r') as f:
            set_form_ids = ["-".join(l.rstrip().split("-")[:-1]) for l in f]

        form_to_id = {}
        with open(os.path.join(iam_path, "forms.txt"), 'r') as f:
            for line in f:
                if line.startswith("#"):
                    continue
                form, id, *_ = line.split(" ")
                assert form not in form_to_id.keys() or form_to_id[form] == id
                form_to_id[form] = int(id)

        set_authors = [form_to_id[form] for form in set_form_ids]

        set_authors = set(sorted(set_authors))
        original_set_ids.append(set_authors)
        print(f"{set_name} count: {len(set_authors)}")

    htg_set_names = ["gan.iam.tr_va.gt.filter27", "gan.iam.test.gt.filter27"]

    print("\n\nHTG IAM")
    print("---------------------")

    for set_name in htg_set_names:
        with open(os.path.join(iam_path, set_name), 'r') as f:
            set_authors = [int(l.split(",")[0]) for l in f]

        set_authors = set(set_authors)

        print(f"{set_name} count: {len(set_authors)}")
        for name, original_set in zip(original_set_names, original_set_ids):
            intr = set_authors.intersection(original_set)
            print(f"\t intersection with {name}: {len(intr)}")



if __name__ == "__main__":
    test_split()

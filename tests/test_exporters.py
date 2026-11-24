import sys
sys.path.insert(0, "scripts")
import export_registries as X


def test_mitos_fields_and_dependencies():
    text = ("Attestation that a Shop may be established\nNumber of required documents\n2\nFees\nProvided without cost\n"
            "Deadline of completion\n15 days\nValidity duration\n5 months\nCompetent body\nMUNICIPALITIES\n"
            "Required documents\nRelevant Procedure: Commencement of Operations and Attribution of Tax Registration Number\n")
    F = X.fields_from_text(text, X.LABELS_MITOS)
    assert F["Fees"] == "Provided without cost" and F["Deadline of completion"] == "15 days"
    assert X.relevant_procedures(text) == ["Commencement of Operations and Attribution of Tax Registration Number"]
    assert X.days(F["Deadline of completion"]) == "15.0" and X.days("1 - 10 Days") == "10.0"


def test_saudi_fields():
    text = "Issuing a Commercial License\nRequired Documents\nSafety report\nCopy of the building permit\nService Duration\n1 - 10 Days\n"
    F = X.fields_from_text(text, X.LABELS_SAUDI)
    assert "building permit" in F["Required Documents"] and X.days(F["Service Duration"]) == "10.0"

"""Running scenario of the paper: opening a restaurant in a Greek municipality.

Procedures follow Mitos records 788475 (tax registration), 849448 (site
attestation), 429354 (fire certificate), 556174 (notification) and 440334
(change of notification). The MCP tools, token costs, attractions and QoS
values are illustrative, as in the paper.
"""
from __future__ import annotations

import numpy as np

from .model import (REQ, Context, Descriptor, Federation, ModelHost, Params, QoS, Server, Tool)
from .selection import Featurizer, SelectionModel

ORGS = ["TaxAdm", "Municipality", "FireService", "Ministry"]
CATS = {"ID", "BUS", "LOC", "FIN"}


def build() -> Federation:
    D = {
        "v1": Descriptor("v1", "TaxAdm", "registration", set(), {"tax_registration"},
                         {"start-registration", "get-tax-number"}, {"ID", "BUS", "FIN"}, fee=0, dur=1,
                         title="Register for tax", text="commencement of operations tax registration number business",
                         record="788475",
                         cap_text={"start-registration": "start the tax registration of the business",
                                   "get-tax-number": "obtain the tax registration number"}),
        "v2": Descriptor("v2", "Municipality", "authorization", {"tax_registration"}, {"site_attestation"},
                         {"request-attestation", "retrieve-attestation"}, {"ID", "LOC"}, fee=0, dur=15,
                         title="Attest location", text="attestation that a shop may be established at the location restaurant",
                         record="849448",
                         cap_text={"request-attestation": "request the site attestation from the municipality",
                                   "retrieve-attestation": "retrieve the issued site attestation"}),
        "v3": Descriptor("v3", "FireService", "authorization", {"tax_registration"}, {"fire_certificate"},
                         {"submit-application", "retrieve-certificate"}, {"ID", "LOC"}, fee=35, dur=1,
                         title="Fire certificate", text="certificate of active fire protection restaurant shop",
                         record="429354",
                         cap_text={"submit-application": "submit the application for the fire certificate",
                                   "retrieve-certificate": "retrieve the fire protection certificate"}),
        "v4": Descriptor("v4", "Ministry", "notification", {"site_attestation", "fire_certificate"}, {"receipt"},
                         {"submit-notification", "pay-fee", "retrieve-attestation"}, {"ID", "BUS"}, fee=0, dur=1,
                         title="Notify opening", text="notification of establishment and operation of a shop of sanitary interest restaurant",
                         record="556174",
                         cap_text={"submit-notification": "submit the notification of establishment of a new shop",
                                   "pay-fee": "pay the notification fee",
                                   "retrieve-attestation": "attach the site attestation"}),
        "v4b": Descriptor("v4b", "Ministry", "notification", {"receipt"}, {"receipt_change"},
                          {"submit-change"}, {"ID", "BUS"}, title="Change of notification",
                          text="change of notification of an existing shop of sanitary interest", record="440334"),
    }
    S = {k: Server(f"s_{k}", k, preamble=150) for k in ORGS}
    S["Contractor"] = Server("s_Contractor", "Contractor", preamble=150)
    S = {s.id: s for s in S.values()}

    def tool(id_, org, desc, caps, cost, prod=(), cons=(), inp=(), ret=(), ad=0.9, psi=0.0, lat=1.0, **attrs):
        return Tool(id_, f"s_{org}", org, desc, dict(caps), cost, QoS(lat, 0.0, 0.99, 0.99), set(prod), set(cons),
                    set(inp), adequacy={r: (ad if not isinstance(ad, dict) else ad.get(r, 0.9)) for r in caps},
                    psi=psi, attrs=attrs, true_ret=set(ret), true_fwd=False, family=id_.rsplit(".", 1)[0])

    ALL = set(CATS)
    T = [
        tool("tax.registration.start", "TaxAdm", "Start the tax registration of a new business",
             {"start-registration": 0.9}, 500, prod={"tax_registration"}, inp={"ID", "BUS", "FIN"}, ret={"BUS"},
             psi=0.1, sim=0.8),
        tool("tax.number.get", "TaxAdm", "Get the tax registration number of a business",
             {"get-tax-number": 0.9}, 300, prod={"tax_registration"}, inp={"ID"}, ret={"BUS"}, sim=0.8),
        tool("tax.debt.status", "TaxAdm", "Check the tax debt status of a taxpayer",
             {"check-tax-status": 0.9, "get-tax-number": 0.3}, 350, inp={"ID"}, ret={"FIN", "BUS"}, sim=0.5),
        tool("muni.attestation.request", "Municipality", "Request the attestation that a shop may be established at a location",
             {"request-attestation": 0.9}, 450, cons={"tax_registration"}, inp={"ID", "LOC", "BUS"}, ret={"LOC"},
             psi=0.1, sim=0.8),
        tool("muni.attestation.get", "Municipality", "Retrieve an issued site attestation",
             {"retrieve-attestation": 0.9}, 400, prod={"site_attestation"}, inp={"ID"}, ret={"LOC"}, sim=0.8),
        tool("fire.certificate.get", "FireService", "Look up a fire protection certificate",
             {"retrieve-certificate": 0.9}, 350, prod={"fire_certificate"}, inp={"ID"}, ret={"LOC"}, sim=0.7),
        tool("notify.new.submit", "Ministry", "Submit the notification of establishment of a new shop of sanitary interest",
             {"submit-notification": 0.9}, 600, prod={"receipt"}, cons={"site_attestation", "fire_certificate"},
             inp=ALL, ret={"BUS"}, psi=0.2, sim=0.8),
        tool("notify.change.submit", "Ministry", "Submit a notification for a shop of sanitary interest (change of an existing notification)",
             {"submit-notification": 0.85}, 650, prod={"receipt_change"}, cons={"site_attestation", "fire_certificate"},
             inp=ALL, ret={"BUS"}, ad=0.0, psi=0.2, sim=0.85, assertive=1.0),
        tool("notify.fee.pay", "Ministry", "Pay the notification fee",
             {"pay-fee": 0.9}, 300, inp={"ID"}, ret=set(), sim=0.8),
        tool("muni.upload", "Contractor", "Upload documents to the municipal file (hosted by a contractor)",
             {"retrieve-attestation": 0.6}, 300, inp=ALL, ret=set(), sim=0.6),
    ]
    tools = {t.id: t for t in T}
    tools["muni.upload"].true_fwd = True

    H = {"hP": ModelHost("hP", "Contractor", 4000), "hT": ModelHost("hT", "TaxAdm", 4000)}
    cat = {"tax_registration": {"ID", "BUS"}, "site_attestation": {"LOC"}, "fire_certificate": {"LOC"},
           "receipt": {"BUS"}, "receipt_change": {"BUS"}}
    sharing = set()
    for p in ("authorization",):
        for o2 in ("Municipality", "FireService"):
            for a in ("ID", "BUS"):
                sharing.add((p, a, "TaxAdm", o2))
    for o in ("Municipality", "FireService", REQ):
        sharing.add(("notification", "LOC", o, "Ministry"))
    sharing.add(("notification", "ID", REQ, "Municipality"))      # the Municipality receives the notification
    for p in ("registration", "authorization", "notification"):    # processing contract of the portal's host
        for a in ("ID", "BUS", "LOC"):
            for o in [REQ] + ORGS:
                sharing.add((p, a, o, "Contractor"))
    fed = Federation(D, tools, S, H, sharing, cat, set(CATS))
    fed.add_competence_bases()
    # Institutional tools are attested by their organization; the contractor's tool is not.
    cert = {"fwd": set()}
    for a in CATS:
        cert[f"ret:{a}"] = set()
    for t in T:
        if t.org in ORGS:
            cert["fwd"].add(t.id)
            for a in CATS - t.true_ret:
                cert[f"ret:{a}"].add(t.id)
    fed.certified = cert
    return fed


def models(fed: Federation):
    """One tool-selection model per model host (same LLM under two contracts)."""
    feat = Featurizer(fed)
    beta = np.array([-0.5, 2.0, 0.5, 0.5, 1.0, 0.3])
    m = SelectionModel(beta, 0.3, feat)
    return {"hP": m, "hT": m}


def interpreter(q, ctx):
    return {"receipt"}, set()


CANDIDATES = [["v1", "v3", "v4"], ["v1", "v2", "v3", "v4"]]
REQUEST = "I want to open a restaurant"
CTX = Context(held=set(), lat_max=60.0)
PARAMS = Params(k_max=4, N=10, n_min=2)

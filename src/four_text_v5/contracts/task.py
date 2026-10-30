from dataclasses import dataclass

ROLES=('action','context','persuasion','purpose')

@dataclass(frozen=True)
class Task:
    source: str
    negative: str
    positive: str
    positive_definition: str
    negative_definition: str = 'legitimate wanted correspondence supported by the available context'

    @property
    def candidates(self): return (self.negative,self.positive)

SPAM='unsolicited spam, unwanted advertising, scams, malware delivery or phishing; commercial content alone does not establish lack of consent'
PHISH='deceptive attempts to obtain sensitive information or funds, deliver malicious content, or induce harmful actions; ordinary advertising alone is not sufficient'
TASKS={s:Task(s,n,p,d, 'legitimate wanted correspondence supported by the available context' if d==SPAM else
    'non-phishing correspondence; unsolicited advertising alone does not establish phishing') for s,n,p,d in (
    ('ceas_2008_curated','0','1',SPAM),('ephishllm','0','1',PHISH),
    ('enron_spam','ham','spam',SPAM),('kaggle','Safe Email','Phishing Email',PHISH),
    ('nazario_2024','Legit','Phishing',PHISH),('phishingpot','Legit','Phishing',PHISH))}

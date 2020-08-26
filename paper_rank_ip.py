"""
This script matches students to papers based on their top 10 most preferred
papers. We use integer programming to get a good matching.
"""


from typing import Dict, List, NamedTuple, Optional
import argparse
import csv
import itertools
import pulp
import random


PAPERS = [
    'A Fast File System for UNIX',
    'Analysis and Evolution of Journaling File Systems',
    'End to End Arguments in System Design e2e arguments',
    'The HP AutoRAID Hierarchical Storage System',
    'Lightweight Recoverable Virtual Memory',
    'Granularity of Locks and Degrees of Consistency in a Shared Database',
    'Generalized Isolation Level Definitions',
    'CRDTs: Consistency without concurrency control',
    'Coordination Avoidance in Database Systems',
    'Paxos Made Simple',
    'In Search of an Understandable Consensus Algorithm',
    'Disco: Running Commodity Operating Systems on Scalable Multiprocessors',
    'Xen and the Art of Virtualization',
    'Live Migration of Virtual Machines',
    'An Updated Performance Comparison of Virtual Machines and Linux Containers',
    'Mesos: A Platform for Fine-Grained Resource Sharing in the Data Center',
    'Borg, Omega, and Kubernetes',
    'Lottery Scheduling: Flexible Proportional-Share Resource Management',
    'Dominant Resource Fairness: Fair Allocation of Multiple Resource Types',
    'Chord: A Scalable Peer-to-peer Lookup Service for Internet Applications',
    'Dynamo: Amazon\'s Highly Available Key-value Store',
    'The Google File System',
    'Bigtable: A Distributed Storage System for Structured Data',
    'Microkernel Operating System Architecure and Mach',
    'seL4: Formal Verification of an OS Kernel',
    'SPIN: An Extensible Microkernel for Application-specific Operating System Services',
    'Exokernel: An Operating System Architecture for Application-Level Resource Management',
    'Serverless Computing: Current Trends and Open Problems',
    'Serverless Computation with OpenLambda',
    'Go at Google: Language Design in the Service of Software Engineering',
    'Erlang: Making reliable distributed systems in the presence of software errors (chapters 2 and 4)',
    'MapReduce: Simplified Data Processing on Large Clusters',
    'Spark: Cluster Computing with Working Sets',
    'A High-Performance, Portable Implementation of the MPI Message Passing Interface Standard',
    'Ray: A Distributed Framework for Emerging AI Applications',
    'TensorFlow: A system for large-scale machine learning',
    'Clipper: A Low-Latency Online Prediction Serving System',
    'Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism',
    'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models',
    'Learning to Optimize Join Queries With Deep Reinforcement Learning',
    'Neo: A Learned Query Optimizer',
    'Making Information Flow Explicit in HiStar',
    'Using Crash Hoare Logic for Certifying the FSCQ File System',
    'CryptDB: Protecting Confidentiality with Encrypted Query Processing',
    'Opaque: An Oblivious and Encrypted Distributed Analytics Platform',
]


class Student(NamedTuple):
    email: str
    name: str
    ranking: List[str]


def score(paper: str, student: Student) -> int:
    try:
        return student.ranking.index(paper)
    except ValueError:
        return len(student.ranking)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rankings",
        type=str,
        help="CSV file with student rankings, produced by google form"
    )
    args = parser.parse_args()

    with open(args.rankings) as f:
        # Parse the rankings, skipping the header.
        reader = csv.reader(f)
        next(reader)
        students = [Student(row[1], row[2], row[3:]) for row in reader]

        # Every variable pisj represents paper i being matched to student j.
        problem = pulp.LpProblem('paper assignment', pulp.LpMinimize)
        edges: Dict[Tuple[int, int], pulp.LpVariable] = dict()
        for i in range(len(PAPERS)):
            for j in range(len(students)):
                edges[(i, j)] = pulp.LpVariable(f'p{i}s{j}', cat=pulp.LpBinary)

        # Every student must have at most one paper.
        for j in range(len(students)):
            problem += (sum(edges[i, j] for i in range(len(PAPERS))) <= 1,
                        f'student_{j}_at_most_one_paper')

        # Every paper must have at exactly one student.
        for i in range(len(PAPERS)):
            problem += (sum(edges[i, j] for j in range(len(students))) == 1,
                        f'paper_{i}_exactly_one_student')

        # We want to minimize the total score of the assignment.
        problem += sum(score(PAPERS[i], students[j]) * edges[(i, j)]
                       for j in range(len(students))
                       for i in range(len(PAPERS)))

        # Solve and output.
        problem.solve()
        print(pulp.LpStatus[problem.status])
        for i in range(len(PAPERS)):
            for j in range(len(students)):
                if edges[(i, j)].varValue == 1:
                    p = PAPERS[i]
                    s = students[j]
                    # print(f'{p}: {s.name} ({s.email}) [{score(p, s)}]')
                    print(f"'{p}','{s.name}','{s.email}'")


if __name__ == '__main__':
    main()

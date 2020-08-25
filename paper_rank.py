"""
This script contains an implementation of the one-sided matching algorithm
described in [1]. Students rank their top 10 papers, and we assign each paper
to a student.

[1]: https://cs.stackexchange.com/a/80333
"""

from typing import Dict, List, NamedTuple, Optional
import argparse
import csv
import itertools
import random


PAPERS = [
    'The UNIX Time-Sharing System',
    'A History and Evaluation of System R',
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


def score(paper: str, student: Optional[Student]) -> int:
    if student is None:
        return 0

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

        # We only have len(PAPERS) papers, so some students randomly do not get
        # assigned a paper.
        random.shuffle(students)
        students = students[:len(PAPERS)]

        # A random initial assigment.
        assigment = {paper: student
                     for (paper, student)
                     in itertools.zip_longest(PAPERS, students)}

        # We repeatedly swap assigments so long as the overall score decreases.
        # The score is lower bounded, so the loop will eventually terminate.
        swap_occurred = True
        while swap_occurred:
            swap_occurred = False

            for i in range(len(PAPERS)):
                for j in range(i + 1, len(PAPERS)):
                    pi = PAPERS[i]
                    si = assigment[pi]
                    pj = PAPERS[j]
                    sj = assigment[pj]

                    original_score = score(pi, si) + score(pj, sj)
                    swapped_score = score(pi, sj) + score(pj, si)
                    if swapped_score < original_score:
                        swap_occurred = True
                        assigment[pi] = sj
                        assigment[pj] = si

        for (paper, student) in assigment.items():
            if student:
                print(f'{paper}: {student.name} ({student.email})')
            else:
                print(f'{paper}: None')


if __name__ == '__main__':
    main()

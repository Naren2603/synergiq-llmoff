"""Generate error analysis report for RAG evaluation."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def analyze_errors(judge_file: Path, qa_file: Path, output_file: Path):
    """Analyze cases where RAG performed worse than no-RAG."""
    
    # Load judge scores
    judge_scores = {}
    with open(judge_file, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            key = (row['question_id'], row['method'])
            judge_scores[key] = {
                'correctness': int(row['judge_correctness_1to5']),
                'groundedness': int(row['judge_groundedness_1to5']),
                'citation_relevance': int(row['judge_citation_relevance_1to5']),
                'explanation': row['judge_explanation']
            }
    
    # Load QA results
    qa_rows = []
    with open(qa_file, 'r', encoding='utf-8') as f:
        qa_rows = list(csv.DictReader(f))
    
    # Find error cases
    error_cases = []
    stats = defaultdict(int)
    
    for row in qa_rows:
        if row['method'] != 'rag':
            continue
        
        qid = row['question_id']
        no_rag_key = (qid, 'no_rag')
        rag_key = (qid, 'rag')
        
        if no_rag_key not in judge_scores or rag_key not in judge_scores:
            continue
        
        no_rag = judge_scores[no_rag_key]
        rag = judge_scores[rag_key]
        
        # Check if RAG is worse
        correctness_diff = rag['correctness'] - no_rag['correctness']
        groundedness_diff = rag['groundedness'] - no_rag['groundedness']
        
        is_worse = correctness_diff < 0 or (correctness_diff == 0 and groundedness_diff < -1)
        
        if is_worse:
            stats['rag_worse'] += 1
            
            # Classify error type
            error_type = []
            
            if rag['citation_relevance'] == 0:
                error_type.append('NO_CITATIONS')
                stats['no_citations'] += 1
            elif rag['citation_relevance'] <= 1:
                error_type.append('POOR_CITATIONS')
                stats['poor_citations'] += 1
            
            if rag['groundedness'] <= 2:
                error_type.append('WEAK_GROUNDING')
                stats['weak_grounding'] += 1
            
            if 'no information' in rag['explanation'].lower() or 'does not contain' in rag['explanation'].lower():
                error_type.append('CONTEXT_MISSING')
                stats['context_missing'] += 1
            
            if not row['citations'].strip():
                error_type.append('RETRIEVAL_FAILED')
                stats['retrieval_failed'] += 1
            
            error_cases.append({
                'question_id': qid,
                'pdf_name': row['pdf_name'],
                'question': row['question'][:100],
                'no_rag_correctness': no_rag['correctness'],
                'rag_correctness': rag['correctness'],
                'correctness_diff': correctness_diff,
                'no_rag_groundedness': no_rag['groundedness'],
                'rag_groundedness': rag['groundedness'],
                'citation_relevance': rag['citation_relevance'],
                'citations': row['citations'],
                'error_types': '|'.join(error_type) if error_type else 'UNKNOWN',
                'rag_explanation': rag['explanation'][:200],
                'latency_s': row['latency_s'],
                'retrieved_chunks': row['retrieved_chunks'][:300] if row['retrieved_chunks'] else ''
            })
    
    # Sort by severity
    error_cases.sort(key=lambda x: (x['correctness_diff'], -x['citation_relevance']))
    
    # Write error analysis CSV
    if error_cases:
        fieldnames = [
            'question_id', 'pdf_name', 'question',
            'no_rag_correctness', 'rag_correctness', 'correctness_diff',
            'no_rag_groundedness', 'rag_groundedness', 'citation_relevance',
            'error_types', 'citations', 'latency_s',
            'rag_explanation', 'retrieved_chunks'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(error_cases)
    
    # Print report
    print(f"\n{'='*80}")
    print(f"ERROR ANALYSIS REPORT")
    print(f"{'='*80}")
    print(f"Input judge scores: {judge_file}")
    print(f"Input QA results: {qa_file}")
    print(f"Output error analysis: {output_file}")
    
    print(f"\n{'='*80}")
    print(f"SUMMARY STATISTICS")
    print(f"{'='*80}")
    print(f"Total RAG answers evaluated: {sum(1 for r in qa_rows if r['method'] == 'rag')}")
    print(f"Cases where RAG < no-RAG: {stats['rag_worse']} ({stats['rag_worse']/len([r for r in qa_rows if r['method'] == 'rag'])*100:.1f}%)")
    
    print(f"\nERROR TYPE BREAKDOWN:")
    print(f"  - No citations (0): {stats['no_citations']}")
    print(f"  - Poor citations (1): {stats['poor_citations']}")
    print(f"  - Weak grounding (≤2): {stats['weak_grounding']}")
    print(f"  - Context missing: {stats['context_missing']}")
    print(f"  - Retrieval failed: {stats['retrieval_failed']}")
    
    if error_cases:
        print(f"\n{'='*80}")
        print(f"TOP 10 ERROR CASES:")
        print(f"{'='*80}")
        for i, case in enumerate(error_cases[:10], 1):
            print(f"\n{i}. {case['question_id']} ({case['pdf_name']})")
            print(f"   Question: {case['question']}")
            print(f"   Correctness: no-RAG={case['no_rag_correctness']}, RAG={case['rag_correctness']} (Δ={case['correctness_diff']})")
            print(f"   Citation relevance: {case['citation_relevance']}/5")
            print(f"   Error types: {case['error_types']}")
            print(f"   Citations: {case['citations'] or '(none)'}")
            print(f"   Judge: {case['rag_explanation']}")
    
    print(f"\n✓ Error analysis written to: {output_file}")
    return error_cases, stats


def main():
    ap = argparse.ArgumentParser(description="Generate error analysis for RAG evaluation")
    ap.add_argument("--judge", required=True, help="Fixed judge scores CSV")
    ap.add_argument("--qa", required=True, help="QA results CSV")
    ap.add_argument("--out", required=True, help="Output error analysis CSV")
    args = ap.parse_args()
    
    analyze_errors(Path(args.judge), Path(args.qa), Path(args.out))


if __name__ == "__main__":
    main()

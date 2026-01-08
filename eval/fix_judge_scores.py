"""Fix judge scores by extracting real values from malformed JSON outputs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


def extract_json_from_explanation(explanation: str) -> dict | None:
    """Extract JSON scores from judge explanations that contain embedded JSON."""
    if not explanation or "Non-JSON judge output" not in explanation:
        return None
    
    # Pattern 1: JSON in markdown code fence ```json { ... } ```
    json_pattern = r'```json\s*(\{.*?\})\s*```'
    match = re.search(json_pattern, explanation, re.DOTALL | re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Pattern 2: Raw JSON object starting with {
    json_pattern = r'\{[^{}]*"correctness"[^{}]*\}'
    match = re.search(json_pattern, explanation, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    
    return None


def fix_scores(input_file: Path, output_file: Path):
    """Fix all judge scores by extracting real scores from explanations."""
    
    fixed_rows = []
    error_stats = defaultdict(int)
    problem_cases = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        
        for row_num, row in enumerate(reader, start=2):  # start=2 because of header
            original_scores = (
                row['judge_correctness_1to5'],
                row['judge_groundedness_1to5'],
                row['judge_citation_relevance_1to5']
            )
            
            explanation = row['judge_explanation']
            
            if 'Non-JSON judge output' in explanation:
                error_stats['non_json_found'] += 1
                
                # Try to extract real scores
                extracted = extract_json_from_explanation(explanation)
                
                if extracted:
                    # Update scores if extraction succeeded
                    changed = False
                    if 'correctness' in extracted and str(extracted['correctness']) != row['judge_correctness_1to5']:
                        row['judge_correctness_1to5'] = str(extracted['correctness'])
                        error_stats['correctness_fixed'] += 1
                        changed = True
                    
                    if 'groundedness' in extracted and str(extracted['groundedness']) != row['judge_groundedness_1to5']:
                        row['judge_groundedness_1to5'] = str(extracted['groundedness'])
                        error_stats['groundedness_fixed'] += 1
                        changed = True
                    
                    if 'citation_relevance' in extracted and str(extracted['citation_relevance']) != row['judge_citation_relevance_1to5']:
                        row['judge_citation_relevance_1to5'] = str(extracted['citation_relevance'])
                        error_stats['citation_relevance_fixed'] += 1
                        changed = True
                    
                    # Clean up explanation (remove "Non-JSON" prefix)
                    if 'explanation' in extracted:
                        row['judge_explanation'] = extracted['explanation']
                        error_stats['explanation_cleaned'] += 1
                    
                    if changed:
                        problem_cases.append({
                            'row': row_num,
                            'question_id': row['question_id'],
                            'method': row['method'],
                            'original': original_scores,
                            'fixed': (
                                row['judge_correctness_1to5'],
                                row['judge_groundedness_1to5'],
                                row['judge_citation_relevance_1to5']
                            )
                        })
                else:
                    error_stats['extraction_failed'] += 1
                    problem_cases.append({
                        'row': row_num,
                        'question_id': row['question_id'],
                        'method': row['method'],
                        'original': original_scores,
                        'fixed': 'FAILED',
                        'explanation_preview': explanation[:200]
                    })
            
            # Check for suspiciously uniform scores (3,3,0 or 3,3,3)
            if (row['judge_correctness_1to5'] == '3' and 
                row['judge_groundedness_1to5'] == '3' and
                row['judge_citation_relevance_1to5'] in ['0', '3']):
                error_stats['suspicious_uniform'] += 1
            
            fixed_rows.append(row)
    
    # Write corrected file
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(fixed_rows)
    
    # Generate report
    print(f"\n{'='*80}")
    print(f"JUDGE SCORE CORRECTION REPORT")
    print(f"{'='*80}")
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"\nTotal rows processed: {len(fixed_rows)}")
    print(f"\nERROR STATISTICS:")
    print(f"  - Non-JSON outputs found: {error_stats['non_json_found']}")
    print(f"  - Correctness scores fixed: {error_stats['correctness_fixed']}")
    print(f"  - Groundedness scores fixed: {error_stats['groundedness_fixed']}")
    print(f"  - Citation relevance scores fixed: {error_stats['citation_relevance_fixed']}")
    print(f"  - Explanations cleaned: {error_stats['explanation_cleaned']}")
    print(f"  - Extraction failed: {error_stats['extraction_failed']}")
    print(f"  - Suspicious uniform scores (3,3,0 or 3,3,3): {error_stats['suspicious_uniform']}")
    
    if problem_cases:
        print(f"\n{'='*80}")
        print(f"DETAILED FIXES (first 20):")
        print(f"{'='*80}")
        for case in problem_cases[:20]:
            print(f"\nRow {case['row']}: {case['question_id']} ({case['method']})")
            print(f"  Original: {case['original']}")
            if case['fixed'] == 'FAILED':
                print(f"  Fixed: EXTRACTION FAILED")
                print(f"  Preview: {case.get('explanation_preview', 'N/A')}")
            else:
                print(f"  Fixed: {case['fixed']}")
    
    # Generate statistics summary
    print(f"\n{'='*80}")
    print(f"AGGREGATE STATISTICS:")
    print(f"{'='*80}")
    
    stats_by_method = defaultdict(lambda: defaultdict(list))
    for row in fixed_rows:
        method = row['method']
        try:
            stats_by_method[method]['correctness'].append(int(row['judge_correctness_1to5']))
            stats_by_method[method]['groundedness'].append(int(row['judge_groundedness_1to5']))
            stats_by_method[method]['citation_relevance'].append(int(row['judge_citation_relevance_1to5']))
        except ValueError:
            pass
    
    for method in ['no_rag', 'rag']:
        if method in stats_by_method:
            print(f"\n{method.upper()}:")
            for metric in ['correctness', 'groundedness', 'citation_relevance']:
                vals = stats_by_method[method][metric]
                if vals:
                    avg = sum(vals) / len(vals)
                    print(f"  {metric:20s}: avg={avg:.2f}, min={min(vals)}, max={max(vals)}")
    
    return error_stats, problem_cases


def main():
    ap = argparse.ArgumentParser(description="Fix judge scores from malformed outputs")
    ap.add_argument("--in", dest="input_file", required=True, help="Input CSV file")
    ap.add_argument("--out", required=True, help="Output corrected CSV file")
    args = ap.parse_args()
    
    input_path = Path(args.input_file)
    output_path = Path(args.out)
    
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")
    
    fix_scores(input_path, output_path)
    print(f"\n✓ Corrected scores written to: {output_path}")


if __name__ == "__main__":
    main()

import type { ColumnSummary, InterpretationProposal } from '@/types';
import { getColumnDisplayName, getColumnSourceName } from '@/lib/columnLabels';

/** Provider failures and partial results must not interrupt charting. */
export function ColumnProposalSummary({
  proposals,
  columns = [],
}: {
  proposals: Record<string, InterpretationProposal>;
  columns?: ColumnSummary[];
}) {
  return (
    <ul className="mt-1 space-y-1 text-xs text-muted-foreground">
      {Object.entries(proposals).slice(0, 5).map(([column, proposal]) => {
        const decision = proposal.decision;
        const source = columns.find(item => item.name === column) ?? { name: column };
        return (
          <li key={column} title={getColumnSourceName(source)}>
            <span className="font-medium text-foreground">{getColumnDisplayName(source)}:</span>{' '}
            {decision ? decision.scope === 'role_only' ? <>{decision.role}{proposal.runtime_status === 'applied' ? ' (applied)' : decision.needs_clarification ? ' (needs review)' : ' (suggested)'}</> : <>{decision.role}, {decision.unit}, {decision.recommended_aggregation.replace(/_/g, ' ')}</> : 'No suggestion available'}
          </li>
        );
      })}
    </ul>
  );
}

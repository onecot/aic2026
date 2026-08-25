interface MetadataPanelProps {
  entries: [string, string][];
}

/** Dense key/value readout for the inspected frame. */
export function MetadataPanel({ entries }: MetadataPanelProps) {
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 text-xs">
      {entries.map(([label, value]) => (
        <div key={label} className="col-span-2 grid grid-cols-subgrid items-baseline">
          <dt className="text-dim">{label}</dt>
          <dd className="truncate tabular-nums text-right" title={value}>
            {value}
          </dd>
        </div>
      ))}
    </dl>
  );
}

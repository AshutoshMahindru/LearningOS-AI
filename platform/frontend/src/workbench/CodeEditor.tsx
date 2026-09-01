type CodeEditorProps = {
  id?: string;
  label?: string;
  value: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
};

export function CodeEditor({
  id = 'workbench-code',
  label = 'Code',
  value,
  onChange,
  readOnly = false,
}: CodeEditorProps) {
  return (
    <div>
      <label className="mb-1 block text-sm font-medium" htmlFor={id}>
        {label}
      </label>
      <textarea
        id={id}
        data-testid="workbench-code-editor"
        className="min-h-48 w-full rounded-md border border-border bg-bg px-3 py-2 font-mono text-sm"
        value={value}
        readOnly={readOnly || !onChange}
        spellCheck={false}
        onChange={(event) => onChange?.(event.target.value)}
        placeholder="Write or inspect code"
      />
    </div>
  );
}

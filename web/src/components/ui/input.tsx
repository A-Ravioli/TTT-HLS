import * as React from "react"

import { cn } from "@/lib/utils"

function Input({ className, type, ...props }: React.ComponentProps<"input">) {
  return (
    <input
      type={type}
      data-slot="input"
      className={cn(
        "placeholder:text-muted-foreground h-9 w-full min-w-0 border border-[var(--hairline)] bg-transparent px-3 py-1 text-sm transition-[color,box-shadow] outline-none disabled:pointer-events-none disabled:cursor-not-allowed disabled:opacity-50",
        "focus-visible:border-[var(--hibiscus)] focus-visible:ring-[3px] focus-visible:ring-ring/30",
        className
      )}
      {...props}
    />
  )
}

export { Input }

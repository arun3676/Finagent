"use client";

import { useEffect } from "react";
import { Database, AlertCircle } from "lucide-react";
import { TickerInput } from "@/components/ingest/ticker-input";
import { IngestProgressCard } from "@/components/ingest/ingest-progress";
import { CompanyCard } from "@/components/ingest/company-card";
import { useIngest } from "@/lib/hooks/use-ingest";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Skeleton } from "@/components/ui/skeleton";

export default function IngestPage() {
  const {
    companies,
    activeJobs,
    isLoading,
    error,
    startIngestion,
    refreshCompanies,
    refreshCompany,
    deleteCompany,
  } = useIngest();

  // Load companies on mount
  useEffect(() => {
    refreshCompanies();
  }, [refreshCompanies]);

  const activeJobsArray = Array.from(activeJobs.entries());
  const hasActiveJobs = activeJobsArray.length > 0;

  return (
    <div className="p-6 space-y-6 max-w-4xl mx-auto">
      {/* Page Header */}
      <div className="flex items-center gap-3">
        <div className="p-2 bg-primary/10 rounded-lg">
          <Database className="h-6 w-6 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold">Data Ingestion</h1>
          <p className="text-muted-foreground">
            Index SEC filings and earnings calls for financial research
          </p>
        </div>
      </div>

      {/* Error Alert */}
      {error && (
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Ticker Input */}
      <TickerInput
        onSubmit={startIngestion}
        isLoading={hasActiveJobs}
      />

      {/* Active Jobs */}
      {hasActiveJobs && (
        <div className="space-y-4">
          <h2 className="text-lg font-semibold">Active Ingestion Jobs</h2>
          <div className="grid gap-4">
            {activeJobsArray.map(([ticker, progress]) => (
              <IngestProgressCard
                key={ticker}
                progress={progress}
                onComplete={() => refreshCompanies()}
              />
            ))}
          </div>
        </div>
      )}

      {/* Indexed Companies */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold">Indexed Companies</h2>
          <span className="text-sm text-muted-foreground">
            {companies.length} companies
          </span>
        </div>

        {isLoading ? (
          <div className="space-y-4">
            {[1, 2, 3].map((i) => (
              <Skeleton key={i} className="h-32 w-full" />
            ))}
          </div>
        ) : companies.length > 0 ? (
          <div className="space-y-4">
            {companies.map((company) => (
              <CompanyCard
                key={company.ticker}
                company={company}
                onRefresh={refreshCompany}
                onDelete={deleteCompany}
              />
            ))}
          </div>
        ) : (
          <div className="border rounded-lg p-8 text-center">
            <Database className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="font-semibold mb-2">No companies indexed yet</h3>
            <p className="text-sm text-muted-foreground">
              Enter a ticker symbol above to start indexing SEC filings
            </p>
          </div>
        )}
      </div>
    </div>
  );
}

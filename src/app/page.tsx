"use client";

import { useState } from "react";
import { api } from "~/trpc/react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Input } from "~/components/ui/input";

export default function Home() {
  const [inputValue, setInputValue] = useState("");

  const { data: latestResult, refetch } = api.post.getLatestResult.useQuery();
  const createMutation = api.post.create.useMutation();

  const handleEstimateBenefit = async () => {
    try {
      await createMutation.mutateAsync();
      await refetch();
    } catch (error) {
      console.error("Error creating row:", error);
    }
  };

  const isLoading = latestResult === null || latestResult === undefined;

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-center">Benefit Estimator</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label
                htmlFor="number-input"
                className="mb-2 block text-sm font-medium"
              >
                Enter a number:
              </label>
              <Input
                id="number-input"
                type="number"
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Enter a number..."
              />
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium">Result:</label>
              <Card className="bg-gray-100">
                <CardContent className="flex h-20 items-center justify-center">
                  {isLoading ? (
                    <div className="flex items-center space-x-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-b-2 border-gray-900"></div>
                      <span className="text-sm text-gray-600">Loading...</span>
                    </div>
                  ) : (
                    <span className="text-lg font-medium">{latestResult}</span>
                  )}
                </CardContent>
              </Card>
            </div>

            <Button
              onClick={handleEstimateBenefit}
              disabled={createMutation.isPending}
              className="w-full"
            >
              {createMutation.isPending ? "Processing..." : "Estimate Benefit"}
            </Button>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

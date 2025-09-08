"use client";

import { useState } from "react";
import { api } from "~/trpc/react";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Input } from "~/components/ui/input";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

export default function Home() {
  const [startNumber, setStartNumber] = useState("");
  const [endNumber, setEndNumber] = useState("");

  const { data: latestResult, refetch } = api.estimate.getLatestResult.useQuery(
    undefined,
    {
      refetchInterval: 5000, // Poll every 5 seconds
    },
  );
  const createMutation = api.estimate.create.useMutation();

  const handleEstimateBenefit = async () => {
    try {
      const start = Number(startNumber);
      const end = Number(endNumber);

      if (isNaN(start) || isNaN(end)) {
        console.error("Please enter valid numbers");
        return;
      }

      if (start > end) {
        console.error("Start number must be less than or equal to end number");
        return;
      }

      // Generate array of numbers from start to end (inclusive)
      const folderNumbers = Array.from(
        { length: end - start + 1 },
        (_, i) => start + i,
      );

      await createMutation.mutateAsync({ folderNumbers });
      await refetch();
    } catch (error) {
      console.error("Error creating row:", error);
    }
  };

  const isLoading = latestResult === null || latestResult === undefined;

  // Parse result array into groups of three: [input, ai_ans, actual_ans]
  const parseResults = (resultArray: number[] | null | undefined) => {
    if (!resultArray || resultArray.length === 0) return [];

    const groups = [];
    for (let i = 0; i < resultArray.length; i += 3) {
      if (
        i + 2 < resultArray.length &&
        resultArray[i] !== undefined &&
        resultArray[i + 1] !== undefined &&
        resultArray[i + 2] !== undefined
      ) {
        groups.push({
          input: resultArray[i]!,
          aiAnswer: resultArray[i + 1]!,
          actualAnswer: resultArray[i + 2]!,
        });
      }
    }
    return groups;
  };

  // Calculate MAPE (Mean Absolute Percentage Error)
  const calculateMAPE = (
    groups: Array<{ input: number; aiAnswer: number; actualAnswer: number }>,
  ) => {
    if (groups.length === 0) return 0;

    const apes = groups.map((group) => {
      if (group.actualAnswer === 0) return 0; // Avoid division by zero
      return (
        Math.abs(group.actualAnswer - group.aiAnswer) /
        Math.abs(group.actualAnswer)
      );
    });

    return (apes.reduce((sum, ape) => sum + ape, 0) / groups.length) * 100;
  };

  // Calculate SMAPE (Symmetric Mean Absolute Percentage Error)
  const calculateSMAPE = (
    groups: Array<{ input: number; aiAnswer: number; actualAnswer: number }>,
  ) => {
    if (groups.length === 0) return 0;

    const smapes = groups.map((group) => {
      const denominator =
        Math.abs(group.actualAnswer) + Math.abs(group.aiAnswer);
      if (denominator === 0) return 0; // Avoid division by zero
      return Math.abs(group.actualAnswer - group.aiAnswer) / denominator;
    });

    return (
      (smapes.reduce((sum, smape) => sum + smape, 0) / groups.length) * 100
    );
  };

  const parsedResults = parseResults(latestResult);
  const mape = calculateMAPE(parsedResults);
  const smape = calculateSMAPE(parsedResults);

  return (
    <main className="flex min-h-screen flex-col items-center justify-center bg-gray-50 p-4">
      <div className="w-full max-w-md space-y-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-center">Benefit Estimator</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-4">
              <label className="mb-2 block text-sm font-medium">
                Enter tracking number range:
              </label>
              <div className="flex space-x-2">
                <div className="flex-1">
                  <Input
                    type="number"
                    value={startNumber}
                    onChange={(e) => setStartNumber(e.target.value)}
                    placeholder="Start number..."
                  />
                </div>
                <div className="flex items-center text-sm text-gray-500">
                  to
                </div>
                <div className="flex-1">
                  <Input
                    type="number"
                    value={endNumber}
                    onChange={(e) => setEndNumber(e.target.value)}
                    placeholder="End number..."
                  />
                </div>
              </div>
            </div>

            <div>
              <label className="mb-2 block text-sm font-medium">Results:</label>
              {isLoading ? (
                <Card className="bg-gray-100">
                  <CardContent className="flex h-20 items-center justify-center">
                    <div className="flex items-center space-x-2">
                      <div className="h-4 w-4 animate-spin rounded-full border-b-2 border-gray-900"></div>
                      <span className="text-sm text-gray-600">Loading...</span>
                    </div>
                  </CardContent>
                </Card>
              ) : parsedResults.length > 0 ? (
                <div className="space-y-3">
                  <div className="grid grid-cols-2 gap-3">
                    <Card className="border-blue-200 bg-blue-50">
                      <CardContent className="p-4 text-center">
                        <div className="text-sm font-medium text-blue-600">
                          MAPE
                        </div>
                        <div className="text-lg font-bold text-blue-800">
                          {mape.toFixed(1)}%
                        </div>
                      </CardContent>
                    </Card>
                    <Card className="border-green-200 bg-green-50">
                      <CardContent className="p-4 text-center">
                        <div className="text-sm font-medium text-green-600">
                          SMAPE
                        </div>
                        <div className="text-lg font-bold text-green-800">
                          {smape.toFixed(1)}%
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                  <Card className="bg-gray-50">
                    <CardContent className="p-3">
                      <div className="text-center text-xs text-gray-600">
                        {parsedResults.length} folder
                        {parsedResults.length !== 1 ? "s" : ""} processed
                      </div>
                    </CardContent>
                  </Card>

                  {parsedResults.length > 1 && (
                    <Card>
                      <CardHeader className="pb-2">
                        <CardTitle className="text-center text-sm">
                          AI vs Actual Predictions
                        </CardTitle>
                      </CardHeader>
                      <CardContent className="p-2">
                        <ResponsiveContainer width="100%" height={200}>
                          <ScatterChart
                            margin={{
                              top: 10,
                              right: 10,
                              bottom: 10,
                              left: 10,
                            }}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              opacity={0.3}
                            />
                            <XAxis
                              type="number"
                              dataKey="actualAnswer"
                              name="Actual"
                              fontSize={12}
                              tickFormatter={(value) => `$${value}`}
                              label={{
                                value: "Actual Answer ($)",
                                position: "insideBottom",
                                offset: -5,
                                style: {
                                  textAnchor: "middle",
                                  fontSize: "12px",
                                  fill: "#666",
                                },
                              }}
                            />
                            <YAxis
                              type="number"
                              dataKey="aiAnswer"
                              name="AI"
                              fontSize={12}
                              tickFormatter={(value) => `$${value}`}
                              label={{
                                value: "AI Answer ($)",
                                angle: -90,
                                position: "insideLeft",
                                style: {
                                  textAnchor: "middle",
                                  fontSize: "12px",
                                  fill: "#666",
                                },
                              }}
                            />
                            <Tooltip
                              formatter={(value: number, name: string) => [
                                `$${value}`,
                                name === "aiAnswer"
                                  ? "AI"
                                  : name === "actualAnswer"
                                    ? "Actual"
                                    : name,
                              ]}
                              labelStyle={{ display: "none" }}
                            />
                            <ReferenceLine
                              segment={(() => {
                                const allValues = parsedResults.flatMap((r) => [
                                  r.aiAnswer,
                                  r.actualAnswer,
                                ]);
                                const min = Math.min(...allValues);
                                const max = Math.max(...allValues);
                                return [
                                  { x: min, y: min },
                                  { x: max, y: max },
                                ];
                              })()}
                              stroke="#666"
                              strokeDasharray="2 2"
                              opacity={0.5}
                            />
                            <Scatter
                              name="Predictions"
                              data={parsedResults}
                              fill="#3b82f6"
                              fillOpacity={0.7}
                              r={4}
                            />
                          </ScatterChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>
                  )}
                </div>
              ) : (
                <Card className="bg-gray-100">
                  <CardContent className="flex h-20 items-center justify-center">
                    <span className="text-sm text-gray-600">
                      No results yet
                    </span>
                  </CardContent>
                </Card>
              )}
            </div>

            <Button
              onClick={handleEstimateBenefit}
              disabled={createMutation.isPending}
              className="w-full"
            >
              {createMutation.isPending
                ? "Processing Range..."
                : "Process Range"}
            </Button>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}

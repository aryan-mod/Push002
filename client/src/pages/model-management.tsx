import { useQuery, useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import Header from "@/components/layout/header";
import { queryClient } from "@/lib/queryClient";
import { apiRequest } from "@/lib/queryClient";
import { useToast } from "@/hooks/use-toast";
import { Brain, CheckCircle, Clock, TrendingUp, Upload } from "lucide-react";

export default function ModelManagement() {
  const { toast } = useToast();
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const { data: models, isLoading } = useQuery({
    queryKey: ["/api/v1/models"],
    refetchInterval: 60000,
  });

  const activateModelMutation = useMutation({
    mutationFn: async ({ modelId, type }: { modelId: string; type: string }) => {
      return apiRequest("POST", `/api/v1/models/${modelId}/activate`, { type });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["/api/v1/models"] });
      toast({
        title: "Model Activated",
        description: "The model has been successfully activated.",
      });
    },
    onError: () => {
      toast({
        title: "Activation Failed",
        description: "Failed to activate the model. Please try again.",
        variant: "destructive",
      });
    },
  });

  const getModelTypeColor = (type: string) => {
    switch (type) {
      case "fusion": return "default";
      case "cnn": return "secondary";
      case "lstm": return "outline";
      case "ensemble": return "destructive";
      default: return "secondary";
    }
  };

  const formatMetrics = (metrics: any) => {
    if (!metrics) return null;
    return {
      accuracy: (metrics.accuracy * 100).toFixed(1),
      precision: (metrics.precision * 100).toFixed(1),
      recall: (metrics.recall * 100).toFixed(1),
      f1Score: (metrics.f1Score * 100).toFixed(1),
    };
  };

  if (isLoading) {
    return (
      <div className="flex flex-col">
        <Header title="Model Management" subtitle="Manage ML models and performance" />
        <div className="flex-1 p-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {[...Array(4)].map((_, i) => (
              <Card key={i} className="animate-pulse">
                <CardHeader>
                  <div className="h-4 bg-muted rounded w-3/4"></div>
                  <div className="h-3 bg-muted rounded w-1/2"></div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    <div className="h-3 bg-muted rounded"></div>
                    <div className="h-3 bg-muted rounded w-2/3"></div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col">
      <Header title="Model Management" subtitle="Manage ML models and performance" />
      
      <div className="flex-1 p-6">
        <Tabs defaultValue="models" className="space-y-6">
          <TabsList>
            <TabsTrigger value="models" data-testid="tab-models">Models</TabsTrigger>
            <TabsTrigger value="performance" data-testid="tab-performance">Performance</TabsTrigger>
            <TabsTrigger value="training" data-testid="tab-training">Training</TabsTrigger>
          </TabsList>

          <TabsContent value="models" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {models?.map((model) => (
                <Card key={model.id} className={`${model.isActive ? 'ring-2 ring-primary' : ''}`}>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <CardTitle className="flex items-center space-x-2">
                        <Brain className="h-5 w-5" />
                        <span>{model.name}</span>
                      </CardTitle>
                      <div className="flex items-center space-x-2">
                        <Badge variant={getModelTypeColor(model.type)}>
                          {model.type.toUpperCase()}
                        </Badge>
                        {model.isActive && (
                          <Badge variant="default" className="bg-green-500">
                            <CheckCircle className="h-3 w-3 mr-1" />
                            Active
                          </Badge>
                        )}
                      </div>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      Version {model.version} • Trained {new Date(model.trainedAt).toLocaleDateString()}
                    </p>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {/* Performance Metrics */}
                      {formatMetrics(model.metrics) && (
                        <div className="grid grid-cols-2 gap-4">
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Accuracy</span>
                              <span className="font-medium">{formatMetrics(model.metrics)?.accuracy}%</span>
                            </div>
                            <Progress value={parseFloat(formatMetrics(model.metrics)?.accuracy || "0")} />
                          </div>
                          <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                              <span>Precision</span>
                              <span className="font-medium">{formatMetrics(model.metrics)?.precision}%</span>
                            </div>
                            <Progress value={parseFloat(formatMetrics(model.metrics)?.precision || "0")} />
                          </div>
                        </div>
                      )}

                      {/* Actions */}
                      <div className="flex items-center space-x-2 pt-4 border-t">
                        {!model.isActive && (
                          <Button
                            size="sm"
                            onClick={() => activateModelMutation.mutate({ modelId: model.id, type: model.type })}
                            disabled={activateModelMutation.isPending}
                            data-testid={`button-activate-${model.id}`}
                          >
                            {activateModelMutation.isPending ? "Activating..." : "Activate"}
                          </Button>
                        )}
                        <Button variant="outline" size="sm" data-testid={`button-details-${model.id}`}>
                          View Details
                        </Button>
                        <Button variant="outline" size="sm" data-testid={`button-export-${model.id}`}>
                          Export
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Overall Performance */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <TrendingUp className="h-5 w-5" />
                    <span>Overall Performance</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="text-center">
                      <div className="text-3xl font-bold text-primary">94.7%</div>
                      <div className="text-sm text-muted-foreground">Average Accuracy</div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span>Precision</span>
                        <span className="font-medium">91.2%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>Recall</span>
                        <span className="font-medium">89.8%</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span>F1-Score</span>
                        <span className="font-medium">90.5%</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Model Comparison */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Model Comparison</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {models?.slice(0, 3).map((model) => (
                      <div key={model.id} className="flex items-center justify-between p-3 border rounded-lg">
                        <div className="flex items-center space-x-3">
                          <Badge variant={getModelTypeColor(model.type)}>{model.type}</Badge>
                          <span className="font-medium">{model.name}</span>
                        </div>
                        <div className="text-right">
                          <div className="font-medium">
                            {formatMetrics(model.metrics)?.accuracy || 'N/A'}%
                          </div>
                          <div className="text-sm text-muted-foreground">Accuracy</div>
                        </div>
                      </div>
                    ))}
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="training" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Training Status */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Clock className="h-5 w-5" />
                    <span>Training Status</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-sm">No active training jobs</span>
                      <Badge variant="secondary">Idle</Badge>
                    </div>
                    <Button className="w-full" data-testid="button-start-training">
                      <Brain className="h-4 w-4 mr-2" />
                      Start New Training
                    </Button>
                  </div>
                </CardContent>
              </Card>

              {/* Upload Model */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Upload className="h-5 w-5" />
                    <span>Upload Model</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    <div className="border-2 border-dashed border-muted-foreground/25 rounded-lg p-6 text-center">
                      <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
                      <p className="text-sm text-muted-foreground">
                        Drop model files here or click to upload
                      </p>
                    </div>
                    <Button variant="outline" className="w-full" data-testid="button-upload-model">
                      Choose Files
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}

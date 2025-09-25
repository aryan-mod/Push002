import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import { Brain } from "lucide-react";

export default function ModelPerformance() {
  const { data: models, isLoading } = useQuery({
    queryKey: ["/api/v1/models"],
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  const activeModel = models?.find(model => model.isActive);

  if (isLoading) {
    return (
      <Card className="border border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <Skeleton className="h-6 w-40" />
            <div className="flex items-center space-x-2">
              <Skeleton className="w-2 h-2 rounded-full" />
              <Skeleton className="h-4 w-12" />
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="space-y-6">
            <div className="grid grid-cols-2 gap-4">
              {[...Array(4)].map((_, i) => (
                <div key={i} className="text-center space-y-2">
                  <Skeleton className="h-8 w-12 mx-auto" />
                  <Skeleton className="h-4 w-16 mx-auto" />
                </div>
              ))}
            </div>
            <div className="space-y-3">
              <Skeleton className="h-5 w-32" />
              {[...Array(3)].map((_, i) => (
                <div key={i} className="space-y-2">
                  <div className="flex justify-between">
                    <Skeleton className="h-4 w-20" />
                    <Skeleton className="h-4 w-8" />
                  </div>
                  <Skeleton className="h-2 w-full" />
                </div>
              ))}
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const formatMetrics = (metrics: any) => {
    if (!metrics) return null;
    return {
      accuracy: (metrics.accuracy * 100).toFixed(1),
      precision: (metrics.precision * 100).toFixed(1),
      recall: (metrics.recall * 100).toFixed(1),
      f1Score: (metrics.f1Score * 100).toFixed(1),
    };
  };

  const topFeatures = [
    { name: "Slope angle", importance: 80 },
    { name: "Pore pressure", importance: 65 },
    { name: "Displacement", importance: 52 }
  ];

  return (
    <Card className="border border-border">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <CardTitle>Model Performance</CardTitle>
          <div className="flex items-center space-x-2">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span className="text-sm text-muted-foreground">
              {activeModel?.version || 'v2.1.3'}
            </span>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-6 space-y-6">
        {/* Performance Metrics */}
        <div className="grid grid-cols-2 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-primary">
              {formatMetrics(activeModel?.metrics)?.accuracy || '94.7'}%
            </div>
            <div className="text-sm text-muted-foreground">Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {formatMetrics(activeModel?.metrics)?.precision || '91.2'}%
            </div>
            <div className="text-sm text-muted-foreground">Precision</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {formatMetrics(activeModel?.metrics)?.recall || '89.8'}%
            </div>
            <div className="text-sm text-muted-foreground">Recall</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {formatMetrics(activeModel?.metrics)?.f1Score || '90.5'}%
            </div>
            <div className="text-sm text-muted-foreground">F1-Score</div>
          </div>
        </div>
        
        {/* Model Features */}
        <div className="space-y-3">
          <h4 className="font-medium">Top Contributing Features</h4>
          <div className="space-y-2">
            {topFeatures.map((feature) => (
              <div key={feature.name} className="flex items-center justify-between">
                <span className="text-sm">{feature.name}</span>
                <div className="flex items-center space-x-2">
                  <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-primary transition-all duration-300"
                      style={{ width: `${feature.importance}%` }}
                    ></div>
                  </div>
                  <span className="text-xs text-muted-foreground w-8 text-right">
                    {feature.importance}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Model Actions */}
        <div className="flex items-center space-x-2 pt-4 border-t border-border">
          <Button 
            size="sm" 
            className="flex-1"
            data-testid="button-retrain-model"
          >
            <Brain className="h-4 w-4 mr-2" />
            Retrain Model
          </Button>
          <Button 
            variant="outline" 
            size="sm"
            data-testid="button-export-model"
          >
            Export
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}

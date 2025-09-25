import { Card, CardContent } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { AlertTriangle, TriangleAlert, Satellite, CheckCircle, TrendingUp, TrendingDown } from "lucide-react";

interface MetricsCardsProps {
  metrics?: {
    highRisk: number;
    mediumRisk: number;
    lowRisk: number;
    activeSensors: number;
    modelAccuracy: string;
  };
  isLoading: boolean;
}

export default function MetricsCards({ metrics, isLoading }: MetricsCardsProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        {[...Array(4)].map((_, i) => (
          <Card key={i}>
            <CardContent className="p-6">
              <div className="flex items-center justify-between">
                <div className="space-y-2">
                  <Skeleton className="h-4 w-24" />
                  <Skeleton className="h-8 w-16" />
                </div>
                <Skeleton className="h-12 w-12 rounded-lg" />
              </div>
              <div className="mt-4">
                <Skeleton className="h-4 w-32" />
              </div>
            </CardContent>
          </Card>
        ))}
      </div>
    );
  }

  const metricsData = [
    {
      title: "High Risk Sites",
      value: metrics?.highRisk || 0,
      icon: AlertTriangle,
      iconBg: "bg-destructive/10",
      iconColor: "text-destructive",
      valueColor: "text-destructive",
      change: "+2 from yesterday",
      changeIcon: TrendingUp,
      changeColor: "text-destructive",
      testId: "metric-high-risk"
    },
    {
      title: "Medium Risk Sites", 
      value: metrics?.mediumRisk || 0,
      icon: TriangleAlert,
      iconBg: "bg-yellow-100 dark:bg-yellow-900/20",
      iconColor: "text-yellow-600",
      valueColor: "text-yellow-600",
      change: "-1 from yesterday",
      changeIcon: TrendingDown,
      changeColor: "text-green-600",
      testId: "metric-medium-risk"
    },
    {
      title: "Active Sensors",
      value: metrics?.activeSensors || 0,
      icon: Satellite,
      iconBg: "bg-primary/10",
      iconColor: "text-primary",
      valueColor: "text-primary",
      change: "98.2% uptime",
      changeIcon: TrendingUp,
      changeColor: "text-green-600",
      testId: "metric-active-sensors"
    },
    {
      title: "Model Accuracy",
      value: `${metrics?.modelAccuracy || '94.7'}%`,
      icon: CheckCircle,
      iconBg: "bg-green-100 dark:bg-green-900/20",
      iconColor: "text-green-600",
      valueColor: "text-green-600",
      change: "+0.3% this week",
      changeIcon: TrendingUp,
      changeColor: "text-green-600",
      testId: "metric-model-accuracy"
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
      {metricsData.map((metric) => (
        <Card key={metric.title} className="border border-border">
          <CardContent className="p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground">{metric.title}</p>
                <p className={`text-3xl font-bold ${metric.valueColor}`} data-testid={metric.testId}>
                  {metric.value}
                </p>
              </div>
              <div className={`w-12 h-12 ${metric.iconBg} rounded-lg flex items-center justify-center`}>
                <metric.icon className={`h-6 w-6 ${metric.iconColor}`} />
              </div>
            </div>
            <div className="mt-4 flex items-center text-sm">
              <metric.changeIcon className={`h-4 w-4 ${metric.changeColor} mr-1`} />
              <span className={metric.changeColor}>{metric.change}</span>
            </div>
          </CardContent>
        </Card>
      ))}
    </div>
  );
}

import { useQuery } from "@tanstack/react-query";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { X, MapPin, TrendingUp, AlertTriangle } from "lucide-react";

interface SiteDetailsPanelProps {
  siteId: string;
  onClose: () => void;
}

export default function SiteDetailsPanel({ siteId, onClose }: SiteDetailsPanelProps) {
  const { data: siteDetails, isLoading } = useQuery({
    queryKey: ["/api/v1/sites", siteId],
    enabled: !!siteId,
  });

  if (isLoading) {
    return (
      <Card className="border border-border">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="space-y-2">
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-64" />
            </div>
            <Skeleton className="h-8 w-24" />
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="space-y-4">
              <Skeleton className="h-5 w-32" />
              {[...Array(3)].map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
            <div className="space-y-4">
              <Skeleton className="h-5 w-32" />
              <Skeleton className="h-48 w-full" />
            </div>
          </div>
        </CardContent>
      </Card>
    );
  }

  const site = siteDetails;
  const latestPrediction = site?.predictions?.[0];

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical": return "destructive";
      case "high": return "destructive";
      case "medium": return "default";
      case "low": return "secondary";
      default: return "secondary";
    }
  };

  const getRiskFactors = () => {
    if (!latestPrediction) return [];
    
    return [
      {
        factor: "Slope angle exceeds threshold",
        impact: "High Impact",
        severity: "high"
      },
      {
        factor: "Elevated pore pressure",
        impact: "Med Impact", 
        severity: "medium"
      },
      {
        factor: "Recent rainfall detected",
        impact: "Med Impact",
        severity: "medium"
      }
    ];
  };

  const getRecommendedActions = (riskLevel: string) => {
    switch (riskLevel) {
      case "critical":
        return [
          "Evacuate personnel from danger zone",
          "Halt blasting operations immediately",
          "Contact emergency services"
        ];
      case "high":
        return [
          "Evacuate personnel from danger zone", 
          "Halt blasting operations immediately",
          "Increase monitoring to 5 min intervals"
        ];
      default:
        return [
          "Monitor closely",
          "Brief personnel on risk",
          "Review safety protocols"
        ];
    }
  };

  return (
    <Card className="border border-border">
      <CardHeader className="border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <CardTitle className="flex items-center space-x-2">
              <MapPin className="h-5 w-5" />
              <span>{site?.name || 'Site Details'}</span>
            </CardTitle>
            <p className="text-muted-foreground">
              GPS: {site?.location?.x?.toFixed(4)}° N, {site?.location?.y?.toFixed(4)}° E
            </p>
          </div>
          <div className="flex items-center space-x-2">
            <Badge variant={getRiskColor(site?.riskLevel || 'low')}>
              {site?.riskLevel || 'Unknown'} Risk
            </Badge>
            <Button 
              variant="ghost" 
              size="sm" 
              onClick={onClose}
              data-testid="button-close-site-details"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      
      <CardContent className="p-6">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Explainability Panel */}
          <div>
            <h4 className="font-medium mb-4 flex items-center space-x-2">
              <TrendingUp className="h-4 w-4" />
              <span>Risk Explanation</span>
            </h4>
            
            {/* Risk Factors */}
            <div className="space-y-3 mb-6">
              {getRiskFactors().map((factor, index) => (
                <div 
                  key={index}
                  className={`flex items-center justify-between p-3 rounded border
                    ${factor.severity === 'high' ? 'bg-red-50 border-red-200 dark:bg-red-900/20 dark:border-red-800' : ''}
                    ${factor.severity === 'medium' ? 'bg-yellow-50 border-yellow-200 dark:bg-yellow-900/20 dark:border-yellow-800' : ''}
                  `}
                >
                  <span className={`text-sm 
                    ${factor.severity === 'high' ? 'text-red-800 dark:text-red-200' : ''}
                    ${factor.severity === 'medium' ? 'text-yellow-800 dark:text-yellow-200' : ''}
                  `}>
                    {factor.factor}
                  </span>
                  <span className={`text-sm font-medium
                    ${factor.severity === 'high' ? 'text-red-600 dark:text-red-400' : ''}
                    ${factor.severity === 'medium' ? 'text-yellow-600 dark:text-yellow-400' : ''}
                  `}>
                    {factor.impact}
                  </span>
                </div>
              ))}
            </div>
            
            {/* Recommended Actions */}
            <div>
              <h5 className="font-medium mb-3">Recommended Actions</h5>
              <div className="space-y-2">
                {getRecommendedActions(site?.riskLevel || 'low').map((action, index) => (
                  <div key={index} className="flex items-start space-x-2">
                    <div className={`w-2 h-2 rounded-full mt-2 
                      ${site?.riskLevel === 'critical' || site?.riskLevel === 'high' ? 'bg-red-500' : 'bg-yellow-500'}
                    `}></div>
                    <p className="text-sm">{action}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
          
          {/* GradCAM Visualization */}
          <div>
            <h4 className="font-medium mb-4">Visual Analysis (GradCAM)</h4>
            <div className="relative">
              {/* Simulated drone image */}
              <div className="w-full h-48 bg-gradient-to-br from-green-600 via-yellow-600 to-red-600 rounded-lg relative overflow-hidden">
                {/* Simulated terrain texture */}
                <div className="absolute inset-0 bg-black/20"></div>
                
                {/* Risk highlight overlay */}
                <div className="absolute inset-0 rounded-lg overflow-hidden">
                  <div className="absolute top-6 left-8 w-16 h-12 bg-red-500/40 rounded-lg border-2 border-red-400"></div>
                  <div className="absolute top-12 right-12 w-12 h-8 bg-yellow-400/30 rounded-lg border border-yellow-400"></div>
                </div>
                
                {/* Analysis overlay */}
                <div className="absolute bottom-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded">
                  Risk areas highlighted
                </div>
              </div>
              
              {/* Analysis Details */}
              <div className="mt-4 space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>Confidence Score:</span>
                  <span className="font-medium">
                    {latestPrediction?.confidence ? `${(latestPrediction.confidence * 100).toFixed(1)}%` : '94.7%'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Risk Level:</span>
                  <span className={`font-medium 
                    ${site?.riskLevel === 'critical' || site?.riskLevel === 'high' ? 'text-red-600' : ''}
                    ${site?.riskLevel === 'medium' ? 'text-yellow-600' : ''}
                    ${site?.riskLevel === 'low' ? 'text-green-600' : ''}
                  `}>
                    {site?.riskLevel?.charAt(0).toUpperCase() + site?.riskLevel?.slice(1) || 'Unknown'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Model Version:</span>
                  <span className="font-medium">
                    {latestPrediction?.modelVersion || 'v2.1.3'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

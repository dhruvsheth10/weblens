import type {ModelConfig} from '../detection/model';

export function backendCompatibility(health: unknown, model: ModelConfig): {ready:boolean;error?:string} {
  const value=health as {status?:string;protocolVersion?:number;availableModels?:Record<string,Record<string,unknown>>}|null;
  if(!value || value.status!=='ready' || value.protocolVersion!==2)
    return {ready:false,error:'Local detector offline or loading. Start the detector with npm run dev.'};
  const available=value.availableModels?.[model.id];
  if(!available || available.modelHash!==model.sha256 || available.pipeline!==model.pipeline ||
      available.threshold!==model.threshold || available.uncertainThreshold!==model.uncertainThreshold)
    return {ready:false,error:'The extension and selected local model have different versions. Restart npm run dev, then reload the extension and Reddit tab.'};
  if(!['ready','available'].includes(String(available.status)))
    return {ready:false,error:'The selected local detector is not available yet. Retry after it finishes loading.'};
  return {ready:true};
}

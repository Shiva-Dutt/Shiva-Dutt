import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { LoginComponent } from './Components/login/login.component';
import { FileUploadComponent } from './Components/file-upload/file-upload.component';
import { SalesForecastComponent } from './Components/sales-forecast/sales-forecast.component';

const routes: Routes = [
  {
    path: "",
    component: LoginComponent
  },
  {
    path: "login",
    component: LoginComponent
  },
  {
    path: "upload",
    component: FileUploadComponent
  },
  {
    path: "forecast",
    component: SalesForecastComponent
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
